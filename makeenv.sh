
# pyenv setup
sudo apt update
sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
curl https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"
pyenv update
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.profile
echo 'pyenv activate python3.11.3' >> ~/.profile

# Python virtualenv setup
pyenv install 3.11.3
pyenv virtualenv 3.11.3 python3.11.3
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate python3.11.3

# Python packages
# rsync -azP --exclude=models /home/fanta/workspace/tweets_classification <destination>
pip install --upgrade pip
pip install -r /home/ubuntu/tweets_classification/requirements.txt
