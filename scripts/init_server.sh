
# Update dowloader
sudo apt-get update

# install git
sudo apt-get install git

# install mc
sudo apt-get install mc

# Here you need to create server public key and register it on GitHub


# Git
mkdir /git_repos
cd /git_repos
git init
git pull git@github.com:germandem/ploutos.git
git remote add origin git@github.com:germandem/ploutos.git
git checkout dev


# install pip
sudo apt install python3-pip

# install libs
pip install -r requirements.txt

# Run scrip
python3 source/telegram_bot/bot.py 