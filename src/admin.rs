use std::convert::Infallible;
use std::net::SocketAddr;
use warp::Filter;

/// Start the HTTP admin server for metrics
pub async fn start_server(addr: SocketAddr) {
    info!("Starting metrics server on {}", addr);

    let routes = metrics_endpoint()
        .or(metrics_json_endpoint())
        .or(vars_endpoint())
        .or(vars_json_endpoint());

    warp::serve(routes).run(addr).await;
}

/// GET /metrics - Prometheus/OpenMetrics format
fn metrics_endpoint() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone
{
    warp::path!("metrics")
        .and(warp::get())
        .and_then(prometheus_metrics)
}

/// GET /metrics.json - JSON format (Finagle compatible)
fn metrics_json_endpoint()
-> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("metrics.json")
        .and(warp::get())
        .and_then(json_metrics)
}

/// GET /vars - Human readable format
fn vars_endpoint() -> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("vars").and(warp::get()).and_then(human_metrics)
}

/// GET /vars.json - JSON format (alias)
fn vars_json_endpoint()
-> impl Filter<Extract = (impl warp::Reply,), Error = warp::Rejection> + Clone {
    warp::path!("vars.json")
        .and(warp::get())
        .and_then(json_metrics)
}

async fn prometheus_metrics() -> Result<impl warp::Reply, Infallible> {
    use metriken::Value;
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut lines = Vec::new();
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after UNIX epoch")
        .as_millis();

    for metric in &metriken::metrics() {
        let name = metric.name().replace('/', "_");

        match metric.value() {
            Some(Value::Counter(value)) => {
                if let Some(description) = metric.description() {
                    lines.push(format!(
                        "# TYPE {} counter\n# HELP {} {}\n{} {} {}",
                        name, name, description, name, value, timestamp
                    ));
                } else {
                    lines.push(format!(
                        "# TYPE {} counter\n{} {} {}",
                        name, name, value, timestamp
                    ));
                }
            }
            Some(Value::Gauge(value)) => {
                if let Some(description) = metric.description() {
                    lines.push(format!(
                        "# TYPE {} gauge\n# HELP {} {}\n{} {} {}",
                        name, name, description, name, value, timestamp
                    ));
                } else {
                    lines.push(format!(
                        "# TYPE {} gauge\n{} {} {}",
                        name, name, value, timestamp
                    ));
                }
            }
            Some(Value::Other(other)) => {
                // Handle histograms
                if let Some(histogram) = other.downcast_ref::<metriken::AtomicHistogram>()
                    && let Some(loaded) = histogram.load()
                {
                    // Export common percentiles
                    let percentiles = [50.0, 90.0, 95.0, 99.0, 99.9];
                    if let Ok(Some(values)) = loaded.percentiles(&percentiles) {
                        for (percentile, bucket) in values.iter() {
                            let value = bucket.end();
                            lines.push(format!(
                                "# TYPE {} gauge\n{}{{percentile=\"{}\"}} {} {}",
                                name, name, percentile, value, timestamp
                            ));
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    lines.sort();
    let content = lines.join("\n") + "\n# EOF\n";
    Ok(warp::reply::with_header(
        content,
        "content-type",
        "text/plain; version=0.0.4; charset=utf-8",
    ))
}

async fn json_metrics() -> Result<impl warp::Reply, Infallible> {
    use metriken::Value;
    use serde_json::json;

    let mut metrics = serde_json::Map::new();

    for metric in &metriken::metrics() {
        let name = metric.name();

        match metric.value() {
            Some(Value::Counter(value)) => {
                metrics.insert(name.to_string(), json!(value));
            }
            Some(Value::Gauge(value)) => {
                metrics.insert(name.to_string(), json!(value));
            }
            Some(Value::Other(other)) => {
                // Handle histograms
                if let Some(histogram) = other.downcast_ref::<metriken::AtomicHistogram>()
                    && let Some(loaded) = histogram.load()
                {
                    // Export common percentiles
                    let percentiles = [50.0, 90.0, 95.0, 99.0, 99.9];
                    if let Ok(Some(values)) = loaded.percentiles(&percentiles) {
                        for (percentile, bucket) in values.iter() {
                            let key = format!("{}/p{}", name, (percentile * 10.0) as u32);
                            metrics.insert(key, json!(bucket.end()));
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    Ok(warp::reply::json(&metrics))
}

async fn human_metrics() -> Result<impl warp::Reply, Infallible> {
    use metriken::Value;

    let mut lines = Vec::new();

    for metric in &metriken::metrics() {
        let name = metric.name();

        match metric.value() {
            Some(Value::Counter(value)) => {
                lines.push(format!("{}: {}", name, value));
            }
            Some(Value::Gauge(value)) => {
                lines.push(format!("{}: {}", name, value));
            }
            Some(Value::Other(other)) => {
                // Handle histograms
                if let Some(histogram) = other.downcast_ref::<metriken::AtomicHistogram>()
                    && let Some(loaded) = histogram.load()
                {
                    // Export common percentiles
                    let percentiles = [50.0, 90.0, 95.0, 99.0, 99.9];
                    if let Ok(Some(values)) = loaded.percentiles(&percentiles) {
                        for (percentile, bucket) in values.iter() {
                            lines.push(format!(
                                "{}/p{}: {}",
                                name,
                                (percentile * 10.0) as u32,
                                bucket.end()
                            ));
                        }
                    }
                }
            }
            _ => continue,
        }
    }

    lines.sort();
    let content = lines.join("\n") + "\n";
    Ok(warp::reply::with_header(
        content,
        "content-type",
        "text/plain; charset=utf-8",
    ))
}

use log::info;
