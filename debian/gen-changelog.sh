#!/usr/bin/env bash

VERSION=$(cargo metadata --format-version 1 --no-deps | jq -r '.packages[] | select(.name == "llm-perf") | .version')

cat <<EOM
llm-perf ($VERSION) $(lsb_release -sc); urgency=medium

  * Automated update package for llm-perf $VERSION

 -- Brian Martin <brayniac@gmail.com>  $(date -R)
EOM
