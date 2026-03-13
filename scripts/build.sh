#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
configuration="${CONFIGURATION:-Release}"
derived_data_path="${DERIVED_DATA_PATH:-.build/xcode}"
destination="${DESTINATION:-platform=macOS}"

cd "$root_dir"

xcodebuild build \
  -scheme ZImageCLI \
  -configuration "$configuration" \
  -destination "$destination" \
  -derivedDataPath "$derived_data_path" \
  -skipPackagePluginValidation \
  ENABLE_PLUGIN_PREPAREMLSHADERS=YES \
  CLANG_COVERAGE_MAPPING=NO \
  "$@"

xcodebuild build \
  -scheme ZImageServe \
  -configuration "$configuration" \
  -destination "$destination" \
  -derivedDataPath "$derived_data_path" \
  -skipPackagePluginValidation \
  ENABLE_PLUGIN_PREPAREMLSHADERS=YES \
  CLANG_COVERAGE_MAPPING=NO \
  "$@"
