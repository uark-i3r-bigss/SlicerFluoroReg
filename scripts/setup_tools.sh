#!/usr/bin/env bash
# Explicitly set up frozen pose/annotation runtimes outside the extension installation.
# Usage: bash scripts/setup_tools.sh [--component all|pose|annotation] [tool/settings flags]
# To inspect without installation: python -m fluororeg tools status (with FluoroReg on PYTHONPATH).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pipeline_common.sh"
exec "$py" -m fluororeg tools setup "$@"
