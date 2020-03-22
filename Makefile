GEOPANDAS = ~/Dropbox/g/geopandas
# Linux specific Docker file for mapclassify
IMAGE='sjsrey/mapclassify:2.2'

container:
	docker build -t $(IMAGE) .

# run a shell for our env
cli:
	docker run -it -p 8888:8888 -v ${PWD}:/home/jovyan $(IMAGE) /bin/bash

dev:
	echo ${GEOPANDAS}
	docker run -it -p 8888:8888 -e "PYSALDATA=/home/jovyan/.local/pysal_data" -v ${PWD}:/home/jovyan -v ${GEOPANDAS}:/home/jovyan/geopandas $(IMAGE) sh -c "/home/jovyan/develop.sh && /bin/bash"

term:
	echo ${GEOPANDAS}
	docker run -it -e "PYSALDATA=/home/jovyan/.local/pysal_data" -v ${PWD}:/home/jovyan -v ${GEOPANDAS}:/home/jovyan/geopandas $(IMAGE) sh -c "/home/jovyan/develop.sh && /bin/bash"
