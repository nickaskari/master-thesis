# master-thesis
Master Thesis at TIØ4900 by Nick Askari and Simen Peder Stang.

## Setting up Virtual Environment (we are calling it "packages")

```sh
python -m venv packages 
```
Then activating (MAC OS)..
```sh
source packages/bin/activate
```
For Windows..
```sh
.\packages\Scripts\activate
```
## Handling packages
Add to requirements file after pip installing,
```sh
pip freeze > requirements.txt
```
Installing all libraries from requirements.txt,
```sh
pip install -r requirements.txt
```
