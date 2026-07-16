import xarray as xr
import numpy as np
import pandas as pd
import  geopandas as gpd
from affine import Affine
from datetime import datetime, timedelta
import aiohttp
import asyncio
from aiohttp import ClientTimeout, TCPConnector
import os
from rasterstats import zonal_stats
from scipy.spatial import cKDTree
from tqdm import tqdm
import tempfile
import rasterio
import ssl
from rasterio.transform import from_origin
from dotenv import load_dotenv
import os
import json
import psycopg2 as pg
import os
import tempfile
import logging
from psycopg2.extras import execute_values
import calendar

"""Rodar a partir das 15h, a imagem sai às 13h"""

load_dotenv()

logging.basicConfig(
    filename="log_grib_download.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

db_logger = logging.getLogger("database")
db_handler = logging.FileHandler("log_database_save.log")
db_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
db_logger.addHandler(db_handler)
db_logger.setLevel(logging.INFO)

# conn = pg.connect(
#     dbname=os.environ.get('DATABASE_NAME'), 
#     user=os.environ.get('DATABASE_USER'), 
#     password=os.environ.get('DATABASE_PASSWORD'), 
#     host=os.environ.get('DATABASE_HOST'), 
#     port=os.environ.get('DATABASE_PORT')
# )
# conn.autocommit = False
# cursor = conn.cursor()

def conection_postgres():
    host = os.environ.get('DATABASE_HOST')
    port = os.environ.get('DATABASE_PORT')
    user = os.environ.get('DATABASE_USER')
    password = os.environ.get('DATABASE_PASSWORD')
    database = os.environ.get('DATABASE_NAME')    

    conn = pg.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    return conn.cursor()

def execute_query(query):
    cur = conection_postgres()
    conn = cur.connection
    try:
        cur.execute(query)
        rows = cur.fetchall()
        
        colunas = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=colunas)

        return df

    except Exception as e:
        print(f"Erro ao executar a query: {e}")
        return None

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

async def download_file(url, filename, semaphore):
    try:
        async with semaphore:
            timeout = ClientTimeout(total=60)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            connector = TCPConnector(ssl=ssl_context)

            async with aiohttp.ClientSession(timeout=timeout,connector=connector) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(filename, "wb") as f:
                            while chunk := await response.content.read(1024 * 1024):
                                f.write(chunk)
                        logging.info(f"✅ Download concluído: {filename}")
                        return True
                    else:
                        logging.warning(f"⚠️ Erro HTTP {response.status} para URL: {url}")
                        return False
    except Exception as e:
        logging.exception(f"❌ Erro inesperado ao baixar {url}: {e}")
        return False

async def baixar_grib_hoje(ano, mes, dia):
    hoje = datetime.now() - timedelta(days=1)
    # # hoje = datetime.now()
    # ano, mes, dia = hoje.year, hoje.month, hoje.day 


    url = f"http://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY/{ano}/{mes:02}/MERGE_CPTEC_{ano}{mes:02}{dia:02}.grib2"
    filename = 'grib_dia.grib2'


    # Baixar somente se não existir
    # if not os.path.exists(filename):
    semaphore = asyncio.Semaphore(1)
    sucesso = await download_file(url, filename, semaphore)

    if not sucesso:
            logging.error(f"❌ Falha no download do GRIB para {hoje.strftime('%Y-%m-%d')}")
            return None, hoje


    return filename, hoje



def getCities():
    query = f"SELECT * FROM cities where id != 2"
    df_cities = execute_query(query)

    return df_cities

def getparameters():
    query = f"SELECT * FROM parameters where parameter_type_id = 5"
    df_parameters = execute_query(query)

    return df_parameters


def saveParameter(city_id, dsc):
    sql = f"""
        INSERT INTO parameters (name, parameterizable_type, parameterizable_id, values, created_at, updated_at, parameter_type_id) VALUES ('NewCityParameter', 'City', {city_id}, '{{"climate": {{"dsc": {dsc}}}}}', now(), now(), 5) ON CONFLICT (parameterizable_type, parameterizable_id, parameter_type_id) DO UPDATE SET 
        values = (parameters.values::jsonb || excluded.values::jsonb)::json,
        updated_at = now()
    """
    cur = conection_postgres()
    cur.execute(sql)

    conn = cur.connection
    conn.commit()
    
def createFile(ds):
    res = 0.1

    # Define o transform com base no canto superior esquerdo
    transform = from_origin(
        west=ds.longitude.min().item() - res / 2,
        north=ds.latitude.max().item() + res / 2,
        xsize=res,
        ysize=res
    )

    # Extrair grade do raster
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    coords = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # Criar árvore de busca para coordenadas
    tree = cKDTree(coords)

    # Inicializar lista de resultados
    resultados = []

    data = pd.to_datetime(str(ds.time.values)).date() if "time" in ds.coords else None

    array = ds.prec.values.astype(np.float32)

    # Defina o caminho para salvar o arquivo temporário manualmente
    tmp_path = os.path.join(os.getcwd(), "temp_raster.tif")

    with rasterio.open(
        tmp_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(array, 1)

    return resultados

def createFileAux(ds_total):
    ds_total['longitude'] = (ds_total['longitude'] + 180) % 360 - 180
    ds_total = ds_total.sortby('longitude')

    lats = ds_total['latitude'][::-1].values
    lons = ds_total['longitude'].values

    res_x = (lons.max() - lons.min()) / (len(lons) - 1)
    res_y = (lats.max() - lats.min()) / (len(lats) - 1)
    transform = Affine.translation(lons.min() , lats.max()) * Affine.scale(res_x, -res_y)
    
    output_tiff = os.path.join('', f"current.tiff")

    var_name = list(ds_total.data_vars)[0]  # Pega o nome da primeira variável
    data = ds_total[var_name].values[::-1]  # Obtém os valores da variável
    # ds_total.to_netcdf(output_nc)
    with rasterio.open(output_tiff, 'w', driver='GTiff',
                    height=len(lats), width=len(lons),
                    count=1, dtype=data.dtype,
                    crs='EPSG:4326', transform=transform) as dst:
        dst.write(data, 1)

def calculateZonal(ds):
    municipios_sp = gpd.read_file('zonal/municipios_sp.shp', encoding='utf-8')
    municipios_sp = municipios_sp.to_crs("EPSG:4326")

    latitudes = ds.latitude.values
    longitudes = ds.longitude.values
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    coords = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

    # Criar árvore de busca para coordenadas
    tree = cKDTree(coords)

    # Inicializar lista de resultados
    resultados = []

    data = pd.to_datetime(str(ds.time.values)).date() if "time" in ds.coords else None
    # print(ds)
    # print(ds.data_vars)
    # array = ds.prec.values.astype(np.float32)
    array = ds.rdp.values.astype(np.float32)

    for idx, row in municipios_sp.iterrows():
        # Estatística zonal
        stat = zonal_stats(
            [row['geometry']],
            "current.tiff",
            stats=["max"],
            nodata=np.nan
        )[0]["max"]

        if stat is None or pd.isna(stat):
            centroide = row.geometry.centroid
            dist, idx_nearest = tree.query([centroide.x, centroide.y])
            lat_idx, lon_idx = np.unravel_index(idx_nearest, lon_grid.shape)
            stat = array[lat_idx, lon_idx]

        resultados.append({
            "cd_mun": row["cd_mun"],
            "data": data,
            "prec_max": stat
        })

    return resultados


def verificar_salvamento_hoje():
    arquivo_log = "log_database_save.log"

    if not os.path.exists(arquivo_log):
        return False

    hoje = datetime.now().strftime("%Y-%m-%d")

    with open(arquivo_log, "r", encoding="cp1252") as f:
        linhas = f.readlines()

    for linha in reversed(linhas):
        if "Salvamentos finalizados com sucesso" in linha:
            data_log = linha[:10]

            return data_log == hoje

    return False


def saveHidroData(id, rain_today, date):
    # print(id,rain_today)
    if(rain_today == 0):
        sql = f"""
            INSERT INTO hidroapp_statistics (model_type, model_id, date_hour, dsc, created_at, updated_at) VALUES ('City', {id}, '{date}', 1, now(), now()) ON CONFLICT (model_type, model_id, date_hour) DO UPDATE SET 
            dsc = COALESCE(hidroapp_statistics.dsc,0) + 1,
            updated_at = now()
            RETURNING model_id, dsc
        """

        # cur = conection_postgres()
        # cur.execute(sql)

        # result = cur.fetchone()
        # conn = cur.connection
        # conn.commit()

def save_hidrodata_batch(rows, date):
    """
    rows: lista de tuplas (city_id, rain_today)
    Só insere/atualiza onde rain_today == 0
    """
    filtered = [(city_id,) for city_id, rain_today in rows if rain_today == 0]

    if not filtered:
        return []

    sql = """
        INSERT INTO hidroapp_statistics
            (model_type, model_id, date_hour, dsc, created_at, updated_at)
        VALUES %s
        ON CONFLICT (model_type, model_id, date_hour)
        DO UPDATE SET
            dsc = COALESCE(hidroapp_statistics.dsc, 0) + 1,
            updated_at = now()
        RETURNING model_id, dsc
    """

    template = f"('City', %s, '{date}', 1, now(), now())"

    cur = conection_postgres()
    conn = cur.connection

    execute_values(cur, sql, filtered, template=template, fetch=True)

    conn.commit()
    cur.close()
  

def save_parameters_batch(rows):
    """
    rows: lista de tuplas (city_id, dsc)
    """
    sql = """
        INSERT INTO parameters
            (name, parameterizable_type, parameterizable_id, values, created_at, updated_at, parameter_type_id)
        VALUES %s
        ON CONFLICT (parameterizable_type, parameterizable_id, parameter_type_id)
        DO UPDATE SET
            values = (parameters.values::jsonb || excluded.values::jsonb)::json,
            updated_at = now()
    """

    template = "('NewCityParameter', 'City', %s, %s, now(), now(), 5)"
    values = [
        (city_id, json.dumps({"climate": {"dsc": dsc}}))
        for city_id, dsc in rows
    ]


    cur = conection_postgres()
    conn = cur.connection
    execute_values(cur, sql, values, template=template)
    conn.commit()
    cur.close()

def main():
    pd.set_option('display.max_rows', None)

    if verificar_salvamento_hoje():
        print("Salvamento de hoje já foi realizado.")
    else:
        print("Salvamento de hoje não encontrado. Executando...")
        print('buscando cidades SIBH')
        df_cities = getCities()

        hoje = datetime.now() - timedelta(days=1)
        ano_ref, mes_ref, dia = hoje.year, hoje.month, hoje.day

        """Trecho comentado para permitir a execução retroativa em lote em datas futuras, caso necessário. 
        Todo o restantante do código deve ficar dentro do for dia in range(1, ultimo_dia + 1):""" 
        #=================================================================================================
        # ano_ref = 2026
        # mes_ref = 7
        # dias_no_mes = calendar.monthrange(ano_ref, mes_ref)[1]

        # hoje = datetime.now()

        # # último dia disponível: ontem, e nunca além do fim do mês
        # if ano_ref == hoje.year and mes_ref == hoje.month:
        #     ultimo_dia = (hoje - timedelta(days=1)).day
        # else:
        #     ultimo_dia = dias_no_mes

        # for dia in range(2, ultimo_dia + 1):
           
        parameters = getparameters()
        print(f'baixando raster de chuva para {dia}-{mes_ref}-{ano_ref}')
        filename_hoje, data_hoje = asyncio.run(baixar_grib_hoje(ano_ref, mes_ref, dia))
        print('download finalizado com sucesso')
        ds =  xr.open_dataset(filename_hoje)

        # Define resolução (assume 0.1° que é típico no MERGE, ajuste se necessário)
        # resultados = createFile(ds)

        try:
            createFileAux(ds)
            resultados = calculateZonal(ds)
        finally:
            ds.close()

        df_prec_max = pd.DataFrame(resultados)

        hoje = datetime.now() - timedelta(days=1)
        ano, mes, dia = hoje.year, hoje.month, hoje.day

        ontem = datetime.now() - timedelta(days=2)
        ano_o, mes_o, dia_o = ontem.year, ontem.month, ontem.day

        city_ids = df_cities['id'].unique()
        df_list = []

        for id in city_ids:
            city = df_cities[df_cities['id'] == id].iloc[0]

            parameter = parameters[parameters['parameterizable_id'] == id]

            if(len(parameter) > 0):
                parameter = parameter.iloc[0]
            
                dsc = parameter['values']['climate']['dsc']
                df_list.append({'cd_mun': city['cod_ibge'], 'DSC': dsc })
            
        df_dias_secos = pd.DataFrame(df_list)
        # df_dias_secos = pd.read_csv(f'ds_dsc{ano_o}{mes_o:02}{dia_o:02}.csv')
        # df_dias_secos["cd_mun"] = df_dias_secos["cd_mun"].astype(str)

        df_prec_max["cd_mun"] = df_prec_max["cd_mun"].astype(str)
        df_atualizado = pd.merge(df_dias_secos, df_prec_max[['cd_mun', 'prec_max']], on='cd_mun', how='left')

        # print(df_atualizado)

        def atualizar_dias_secos(row):
            if row['prec_max'] < 1:
                # row['DS'] += 1
                row['DSC'] += 1
                row['rain_today'] = 0
            else:
                # DS mantém, DSC zera
                row['DSC'] = 0
                row['rain_today'] = 1
            return row

        # Aplicar função linha a linha
        df_atualizado = df_atualizado.apply(atualizar_dias_secos, axis=1)
        # df_atualizado['DSC'] = 0 #Para o caso de zerar novamente

        # Atualizar o DataFrame original
        df_dias_secos_new = df_atualizado[['cd_mun',  'DSC', 'rain_today']].copy()

        cds = df_dias_secos_new['cd_mun'].unique()

        df_merged = df_dias_secos_new.merge(df_cities[['cod_ibge', 'id']], left_on='cd_mun', right_on='cod_ibge')

        rows_consec = list(zip(df_merged['id'], df_merged['DSC']))
        rows_today = list(zip(df_merged['id'], df_merged['rain_today']))

        print(f'salvando parametro de dias sem chuva no mes {mes_ref}')
        db_logger.info(f"Salvando parâmetro de dias sem chuva no mês {mes_ref}")
        # save_hidrodata_batch(rows_today, f'{ano_ref}-{mes_ref:02}-{dia:02} 03:00')
        try:
            save_hidrodata_batch(rows_today, f'{ano_ref}-{mes_ref:02}-{dia:02} 03:00')
        except Exception as e:
            db_logger.error(f"Erro ao salvar hidrodata: {e}")
        
        print('salvando parametro de dias sem chuva consecutivos')
        db_logger.info("Salvando parâmetro de dias sem chuva consecutivos")
        # save_parameters_batch(rows_consec)
        try:
            save_parameters_batch(rows_consec)
        except Exception as e:
            db_logger.error(f"Erro ao salvar parameters: {e}")
        

        db_logger.info("Salvamentos finalizados com sucesso")
        

        # for ibge in cds:
        #     dsc = df_dias_secos_new[df_dias_secos_new['cd_mun'] == ibge].iloc[0]['DSC']
        #     rain_today = df_dias_secos_new[df_dias_secos_new['cd_mun'] == ibge].iloc[0]['rain_today']
        #     id = df_cities[df_cities['cod_ibge'] == ibge].iloc[0]['id']
        #     print(ibge)
            
        #     # print(f'salvando {ibge} {id} {dsc}')
        #     # print('salvando parametro de dias sem chuva consecutivos')
        #     # saveParameter(id, dsc)
        #     print('salvando parametro de dias sem chuva no mes 04')
        #     saveHidroData(id, rain_today, '2026-04-01 03:00')


        # df_dias_secos_new.to_csv(f'ds_dsc{ano}{mes:02}{dia:02}.csv', index=False)

main()

