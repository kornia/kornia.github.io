#!/bin/bash -v

# install third-party dependencies
sudo apt-get install ruby-full build-essential zlib1g-dev docker-compose

# setup bash
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# install jekyll
gem install jekyll bundler
