# Cube in a Box

The Cube in a Box is a simple way to run the [Open Data Cube](https://www.opendatacube.org).

## How to use:

### 1. Setup:

**Checkout the Repo:**
> `git clone https://github.com/martinp95/cube-in-a-box.git` or `git clone git@github.com:martinp95/cube-in-a-box.git`

**First time users of Docker should run:**
* `bash setup.sh` - This will get your system running and install everything you need.
* Note that after this step you will either need to logout/login, or run the next step with `sudo`

**If you already have `make` , `docker` and `docker-compose` installed. For a custom bounding box append `BBOX=<left>,<bottom>,<right>,<top>` to the end of the command.**
* `make setup`
* Custom bounding box: `make setup BBOX=-2,37,15,47` or `make setup-prod BBOX=-2,37,15,47`

**If you do not have `make` installed and would rather run the commands individually run the following:**

* Build a local environment: `docker-compose build`
* Start a local environment: `docker-compose up`
* Set up your local postgres database (after the above has finished) using:
  * `docker-compose exec jupyter datacube -v system init`
  * `docker-compose exec jupyter datacube product add https://raw.githubusercontent.com/digitalearthafrica/config/master/products/esa_s2_l2a.odc-product.yaml`
* Index a default region with:
  * `docker-compose exec jupyter bash -c "stac-to-dc --bbox='25,20,35,30' --collections='sentinel-s2-l2a-cogs' --datetime='2020-01-01/2020-03-31'"`
* Shutdown your local environment:
  * `docker-compose down`

### 2. Usage:
View the Jupyter notebook `Sentinel_2.ipynb` at [http://localhost](http://localhost) using the password `secret`. Note that you can index additional areas using the `Indexing_More_Data.ipynb` notebook.
