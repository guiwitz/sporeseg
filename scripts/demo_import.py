import requests
import os

if not os.path.isdir('Sporefolder/SporesA'):
    os.makedirs('Sporefolder/SporesA')
if not os.path.isdir('Sporefolder/SporesB'):
    os.makedirs('Sporefolder/SporesB')

myfile = requests.get('https://www.dropbox.com/s/e1y882oawwxmtn6/spore1.jpg?dl=1', allow_redirects=True)
open('Sporefolder/SporesA/spore1.jpg', 'wb').write(myfile.content)

myfile = requests.get('https://www.dropbox.com/s/f7yufnypmbxfu06/spore2.jpg?dl=1', allow_redirects=True)
open('Sporefolder/SporesA/spore2.jpg', 'wb').write(myfile.content)

myfile = requests.get('https://www.dropbox.com/s/e1y882oawwxmtn6/spore1.jpg?dl=1', allow_redirects=True)
open('Sporefolder/SporesB/spore1.jpg', 'wb').write(myfile.content)

myfile = requests.get('https://www.dropbox.com/s/f7yufnypmbxfu06/spore2.jpg?dl=1', allow_redirects=True)
open('Sporefolder/SporesB/spore2.jpg', 'wb').write(myfile.content)
