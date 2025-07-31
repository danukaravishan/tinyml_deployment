#!/usr/bin/env bash
set -e

echo "Step 1: Checking for pyenv..."
# 1. Install pyenv if missing
if ! command -v pyenv >/dev/null 2>&1; then
  echo "Installing pyenv..."
  curl https://pyenv.run | bash
fi

echo "Step 2: Configuring pyenv in ~/.bashrc if not already present..."
# 2. Add pyenv init to ~/.bashrc if not already present
if ! grep -q 'PYENV_ROOT' ~/.bashrc; then
  echo "Writing pyenv initialization to ~/.bashrc..."
  cat << 'EOF' >> ~/.bashrc
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
EOF
fi

echo "Step 3: Reloading shell configuration..."
# 3. Reload shell configuration
source ~/.bashrc

echo "Step 4: Installing Python 3.9.23 via pyenv..."
# 4. Install Python 3.9.23 via pyenv
pyenv install -s 3.9.23


echo "Step 5: Setting local Python version to 3.9.23..."
# 6. Set local Python version
pyenv local 3.9.23

echo "Step 6: Installing dependencies..."
# 7. Install dependencies
pip install tflite-runtime==2.13.0
pip install --no-binary :all: numpy==1.23.5
pip install psutil

echo "Step 6: Creating logs directory..."
mkdir -p logs

echo "Step 7: Making run.sh executable..."
# 10. Make the run script executable
chmod +x run.sh

echo "Step 8: Setting timezone to Pacific/Auckland..."
sudo timedatectl set-ntp true
sudo timedatectl set-timezone Pacific/Auckland

echo "Step 9: Setup complete. Run './run.sh' to start the application in the background."