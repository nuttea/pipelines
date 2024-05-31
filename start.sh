#!/usr/bin/env bash
PORT="${PORT:-9099}"
HOST="${HOST:-0.0.0.0}"

# Function to install requirements if requirements.txt is provided
install_requirements() {
  if [[ -f "$1" ]]; then
    echo "requirements.txt found at $1. Installing requirements..."
    pip install -r "$1"
  else
    echo "requirements.txt not found at $1. Skipping installation of requirements."
  fi
}

# Check if the PIPELINES_REQUIREMENTS_PATH environment variable is set and non-empty
if [[ -n "$PIPELINES_REQUIREMENTS_PATH" ]]; then
  # Install requirements from the specified requirements.txt
  install_requirements "$PIPELINES_REQUIREMENTS_PATH"
else
  echo "PIPELINES_REQUIREMENTS_PATH not specified. Skipping installation of requirements."
fi


# Function to download the pipeline files
download_pipelines() {
  local path=$1
  local destination=$2

  echo "Downloading pipeline files from $path to $destination..."

  if [[ "$path" =~ ^https://github.com/.*/.*/blob/.* ]]; then
    # It's a single file
    dest_file=$(basename "$path")
    curl -L "$path?raw=true" -o "$destination/$dest_file"
  elif [[ "$path" =~ ^https://github.com/.*/.*/tree/.* ]]; then
    # It's a folder
    git_repo=$(echo "$path" | awk -F '/tree/' '{print $1}')
    subdir=$(echo "$path" | awk -F '/tree/' '{print $2}')
    git clone --depth 1 --filter=blob:none --sparse "$git_repo" "$destination"
    (
      cd "$destination" || exit
      git sparse-checkout set "$subdir"
    )
  else
    echo "Invalid PIPELINES_PATH format."
    exit 1
  fi
}

# Function to parse and install requirements from frontmatter
install_frontmatter_requirements() {
  local file=$1

  echo "Checking $file for requirements in frontmatter..."

  # Extract the frontmatter if it exists
  frontmatter=$(sed -n '/^---$/,/^---$/p' "$file")

  if echo "$frontmatter" | grep -q "requirements:"; then
    requirements=$(echo "$frontmatter" | grep "requirements:" | cut -d ":" -f2- | tr -d ' ')
    echo "Installing requirements: $requirements"
    pip install $(echo $requirements | tr ',' ' ')
  else
    echo "No requirements found in frontmatter of $file."
  fi
}


# Check if PIPELINES_PATH environment variable is set and non-empty
if [[ -n "$PIPELINES_PATH" ]]; then
  pipelines_dir="./pipelines"
  mkdir -p "$pipelines_dir"
  download_pipelines "$PIPELINES_PATH" "$pipelines_dir"

  for file in "$pipelines_dir"/*; do
    if [[ -f "$file" ]]; then
      install_frontmatter_requirements "$file"
    fi
  done
else
  echo "PIPELINES_PATH not specified. Skipping pipelines download and installation."
fi


# Start the server
uvicorn main:app --host "$HOST" --port "$PORT" --forwarded-allow-ips '*'
