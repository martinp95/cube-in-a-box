## You can follow the steps below in order to get yourself a local ODC.
## Start by running `setup` then you should have a system that is fully configured
##
## Once running, you can access a Jupyter environment
## at 'http://localhost' with password 'secret'
.PHONY: help setup up down clean

BBOX := -9.5,36,1.5,43.8

help: ## Print this help
	@grep -E '^##.*$$' $(MAKEFILE_LIST) | cut -c'4-'
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: build up init product index ## Run a full local/development setup

up: ## 1. Bring up your Docker environment
	docker-compose up -d postgres
	docker-compose run checkdb
	docker-compose up -d jupyter

init: ## 2. Prepare the database
	docker-compose exec -T jupyter datacube -v system init

product: ## 3. Add a product definition for Sentinel-2
	docker-compose exec -T jupyter dc-sync-products /conf/products.csv


index: ## 4. Index some data (Change extents with BBOX='<left>,<bottom>,<right>,<top>')
	docker-compose exec -T jupyter bash -c \
		"stac-to-dc \
			--bbox='$(BBOX)' \
			--catalog-href='https://earth-search.aws.element84.com/v0/' \
			--collections='sentinel-s2-l2a-cogs' \
			--datetime='2022-01-01/2022-06-01'"
	
	docker-compose exec -T jupyter bash -c \
		"stac-to-dc \
			--bbox='$(BBOX)' \
			--catalog-href='https://earth-search.aws.element84.com/v0/' \
			--collections='sentinel-s2-l2a-cogs' \
			--datetime='2022-06-01/2022-12-31'"

down: ## Bring down the system
	docker-compose down

build: ## Rebuild the base image
	docker network create odcnet
	docker-compose pull
	docker-compose build

clean: ## Delete everything
	docker network rm odcnet
	docker-compose down --rmi all -v
