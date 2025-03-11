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


Find out what in standard formula is time dependent.

Does not have multi VaR capabilities as of now.

What makes the GAN overly conservetive? - possibly EONIA?

Explainability --> analyse the latent space

1. Add Quartelrly support
2. Make more Conditional GANS --> Equity
3. Make CGAN for Bonds
4. Make Multivarate CGAN and test
7. Investingate qunantile NNs
