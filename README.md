# master-thesis
Master Thesis at TIØ4900 by Nick Askari and Simen Peder Stang.

Code is ran using a homemade distributed system created by us,
Find it at: https://github.com/nickaskari/compute-farm

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

TODO:
1. Fix Json saving and loading
2. Run the GANS, double check plots
3. Add Wasserstein plots
4. Fix backtesting