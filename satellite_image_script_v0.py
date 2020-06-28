# USAGE
# python satellite_image_script.py --user 7x27nHWFRKZhXePiHbVfkHBx9MC3/-M9nlimuyRUhXIlsAicA
## leer librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shapefile
import geopandas as gpd
import folium 
from shapely.geometry import MultiPolygon, Polygon
from sentinelhub import WmsRequest, BBox, CRS, MimeType, CustomUrlParam, get_area_dates
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
from pathlib import Path
from myfunctions import Fire_down
from myfunctions import Cloudless_tools
#import rasterstats as rs
import argparse

## variables dinamicas para correr en terminal unicamente
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--user", required=True,
	help="path of working user/terrain")
args = vars(ap.parse_args())

#main_folder=args["folder"]

#inicializacion de variables - fechas
Date_Ini = '2020-01-01' #replace for dynamic dates
Date_Fin = '2020-01-31' #replace for dynamic dates
Date_Ini_c = Date_Ini.replace('-','')
Date_Fin_c = Date_Fin.replace('-','')

#inicializacion de variables - user / area
user_analysis = (args["user"]) #'7x27nHWFRKZhXePiHbVfkHBx9MC3/-M9nlimuyRUhXIlsAicA'
analysis_area = user_analysis.split("/")[1]
x_width = 768*2    #16km width
y_height = 768*2   #16km height

#funcion para leer firebase
lote_aoi,lote_aoi_wgs,minx,maxx,miny,maxy,bounding_box = Fire_down.find_poly(user_analysis)
print("[INFO] box coordinates (min_x, max_x = {:.2f}, {:.2f})".format(minx,maxx))
print("[INFO] box coordinates (min_y, max_y = {:.2f}, {:.2f})".format(miny,maxy))

#leer shapefile
aoi = gpd.read_file('shapefiles/'+analysis_area+'/big_box.shp') #para imagen satelital
aoi.crs = {'init':'epsg:32618', 'no_defs': True}
aoi_universal= aoi.to_crs(4326)                                 #para API sentinel
footprint = None
for i in aoi_universal['geometry']:                             #area
    footprint = i

#cloud detection
INSTANCE_ID = 'd855383e-2ab0-4f3e-826d-a571630d5dc8' #From Sentinel HUB Python Instance ID /change to dynamic user input
LAYER_NAME = 'TRUE-COLOR-S2-L1C' # e.g. TRUE-COLOR-S2-L1C
#Obtener imagenes por fecha (dentro de rango) dentro de box de interés
wms_true_color_request = WmsRequest(layer=LAYER_NAME,
                                    bbox=bounding_box,
                                    time=(Date_Ini, Date_Fin), #cambiar a fechas de interés
                                    width=x_width, height=y_height,
                                    image_format=MimeType.PNG,
                                    instance_id=INSTANCE_ID)
wms_true_color_imgs = wms_true_color_request.get_data()
#Cloudless_tools.plot_previews(np.asarray(wms_true_color_imgs), wms_true_color_request.get_dates(), cols=4, figsize=(15, 10))

#Calculo de probabilidades y obtención de mascaras de nubes
bands_script = 'return [B01,B02,B04,B05,B08,B8A,B09,B10,B11,B12]'
wms_bands_request = WmsRequest(layer=LAYER_NAME,
                               custom_url_params={
                                   CustomUrlParam.EVALSCRIPT: bands_script,
                                   CustomUrlParam.ATMFILTER: 'NONE'
                               },
                               bbox=bounding_box, 
                               time=(Date_Ini, Date_Fin),
                               width=x_width, height=y_height,
                               image_format=MimeType.TIFF_d32f,
                               instance_id=INSTANCE_ID)
wms_bands = wms_bands_request.get_data()
cloud_detector = S2PixelCloudDetector(threshold=0.35, average_over=8, dilation_size=3) #change threshold to test
cloud_probs = cloud_detector.get_cloud_probability_maps(np.array(wms_bands))
cloud_masks = cloud_detector.get_cloud_masks(np.array(wms_bands))
all_cloud_masks = CloudMaskRequest(ogc_request=wms_bands_request, threshold=0.1)

#folder de imagenes nubes
Path('output_clouds/'+analysis_area).mkdir(parents=True, exist_ok=True)
if not os.path.exists(analysis_area):
    os.makedirs(analysis_area)

#Mostrar las probabilidades de nubes para cada imagen por fecha en el rango de analisis
fig = plt.figure(figsize=(15, 10))
n_cols = 4
n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))
for idx, [prob, mask, data] in enumerate(all_cloud_masks):
    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
    image = wms_true_color_imgs[idx]
    Cloudless_tools.overlay_cloud_mask(image, mask, factor=1, fig=fig)
plt.tight_layout()
plt.savefig('output_clouds/'+analysis_area+'/real_and_cloud.png')
#Mostrar las mascaras de nubes para cada imagen por fecha en el rango de analisis
fig = plt.figure(figsize=(15, 10))
n_cols = 4
n_rows = int(np.ceil(len(wms_true_color_imgs) / n_cols))
for idx, cloud_mask in enumerate(all_cloud_masks.get_cloud_masks(threshold=0.35)): #se repite con linea 101
    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
    Cloudless_tools.plot_cloud_mask(cloud_mask, fig=fig)  
plt.tight_layout()
plt.savefig('output_clouds/'+analysis_area+'/cloud_masks.png')
#Calculo y extracción de imagenes con cobertura de nubes menor a x%
cld_per_idx = []
each_cld_mask = all_cloud_masks.get_cloud_masks(threshold=0.35)                 #se repite con linea 94
for a in range(0,len(each_cld_mask)):
    n_cloud_mask = np.shape(np.concatenate(each_cld_mask[a]))
    cloud_perc = sum(np.concatenate(each_cld_mask[a])== 1)/n_cloud_mask
    cld_per_idx.append(cloud_perc)
x = pd.DataFrame(cld_per_idx)<0.6 #Menor a 60% de cobertura de nubes
valid_dates = pd.DataFrame(all_cloud_masks.get_dates())[x[0]]
print("[INFO] valid dates ... {:f})".format(valid_dates))

#filter clouds dataframe with only valid dates
clouds_data = cloud_masks[x[0]]
minIndex = cld_per_idx.index(min(cld_per_idx))
best_date = valid_dates[valid_dates.index==minIndex]
best_date = best_date.iloc[0,0]

#Mostrar las mascaras de nubes para cada imagen por fecha valida
fig = plt.figure(figsize=(15, 10))
n_cols = 4
n_rows = int(np.ceil(len(clouds_data) / n_cols))
for idx, cloud_mask in enumerate(clouds_data):
    ax = fig.add_subplot(n_rows, n_cols, idx + 1)
    plot_cloud_mask(cloud_mask, fig=fig)
plt.tight_layout()
plt.savefig('output_clouds/'+analysis_area+'/cloud_masks_valid.png')