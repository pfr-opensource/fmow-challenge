import numpy as np
import pandas as pd

cat_impact = pd.Series({'multi-unit_residential': 1.0, 'helipad': 1.0, 'port': 1.0, 'debris_or_rubble': 0.6, 'zoo': 1.0, 'water_treatment_facility': 1.0, 'golf_course': 1.0, 'park': 1.0, 'barn': 1.0, 'tower': 1.4, 'car_dealership': 1.0, 'airport_terminal': 1.0, 'smokestack': 1.4, 'shipyard': 1.0, 'lighthouse': 1.0, 'military_facility': 0.6, 'space_facility': 1.0, 'road_bridge': 1.4, 'parking_lot_or_garage': 1.0, 'stadium': 1.0, 'crop_field': 0.6, 'flooded_road': 0.6, 'recreational_facility': 1.0, 'oil_or_gas_facility': 1.0, 'airport': 0.6, 'burial_site': 1.0, 'construction_site': 1.4, 'place_of_worship': 1.0, 'toll_booth': 1.0, 'interchange': 1.0, 'race_track': 1.0, 'factory_or_powerplant': 1.4, 'storage_tank': 1.0, 'impoverished_settlement': 1.0, 'fire_station': 1.4, 'airport_hangar': 1.0, 'border_checkpoint': 1.4, 'wind_farm': 0.6, 'single-unit_residential': 0.6, 'nuclear_powerplant': 0.6, 'electric_substation': 1.0, 'gas_station': 1.4, 'prison': 1.0, 'aquaculture': 1.0, 'amusement_park': 1.0, 'runway': 1.0, 'lake_or_pond': 1.0, 'tunnel_opening': 0.6, 'swimming_pool': 1.0, 'waste_disposal': 1.0, 'educational_institution': 1.4, 'ground_transportation_station': 1.0, 'surface_mine': 1.0, 'office_building': 1.0, 'police_station': 1.4, 'hospital': 1.0, 'archaeological_site': 1.0, 'shopping_mall': 1.0, 'fountain': 1.0, 'dam': 1.0, 'solar_farm': 0.6, 'railway_bridge': 1.0}).sort_index()

BASELINE_CATEGORIES = pd.Index('false_detection airport airport_hangar airport_terminal amusement_park aquaculture archaeological_site barn border_checkpoint burial_site car_dealership construction_site crop_field dam debris_or_rubble educational_institution electric_substation factory_or_powerplant fire_station flooded_road fountain gas_station golf_course ground_transportation_station helipad hospital interchange lake_or_pond lighthouse military_facility multi-unit_residential nuclear_powerplant office_building oil_or_gas_facility park parking_lot_or_garage place_of_worship police_station port prison race_track railway_bridge recreational_facility impoverished_settlement road_bridge runway shipyard shopping_mall single-unit_residential smokestack solar_farm space_facility stadium storage_tank surface_mine swimming_pool toll_booth tower tunnel_opening waste_disposal water_treatment_facility wind_farm zoo'.split(), name='category')
NEW_MODEL_CATEGORIES = sorted(BASELINE_CATEGORIES)

WIDTHS = pd.Series({'airport': '5000', 'nuclear_powerplant': '5000', 'port': '5000', 'shipyard': '5000', 'amusement_park': '500 1500', 'impoverished_settlement': '1500', 'runway': '1500', 'space_facility': '1500' }, dtype=object).loc[cat_impact.index].fillna('500').map(lambda x: list(map(int, x.split())))
MIN_WIDTHS = WIDTHS.str.get(0)

lerp = lambda t,a,b: (1-t)*a + t*b

def csv_parse(path):
    parsed = pd.read_csv(path, index_col=0, float_precision='roundtrip')
    parsed['timestamp'] = pd.to_datetime(parsed['timestamp'])
    
    is_test = (parsed.mean_pixel_width<0).any()
    if is_test:
        is_ms = parsed.gsd.median() > 1
        if is_ms:
            # Some multispectral images have worse than 4x downsampling
            # and the gsd doesn't take that into account: increase margins.
            gsd_based_threshold = [500+380, 1500+800]
        else:
            gsd_based_threshold = [500+200, 1500+400]
        gsd_based_width = parsed.img_width * parsed.gsd
        width_m = 500 + (gsd_based_width >= gsd_based_threshold[0])*1000 + (gsd_based_width >= gsd_based_threshold[1])*3500
    else:
        width_m = (pd.Series(parsed.img_width * abs(parsed.mean_pixel_width) * (40075/360), name='width_m').round(2) * 1000).astype(int)
    parsed['width_m'] = width_m # the 500/1500/5000 bucket; only an upper bound on physical size
    parsed['size_m'] = width_m * parsed.eval('img_height / img_width') # the physical size of the tile
    
    parsed['crop_id'] = parsed.img_filename.str.extract(r'^(.*)_[^_]*$', expand=False)
    assert parsed['fold'].notnull().all()

    raw_obj_id = parsed.img_filename.str.extract(r'^(\d+|[\w\-]+_\d+)_\d+_(?!\d)', expand=False)
    obj_id = raw_obj_id.str.cat(parsed['fold'], '_') # note: .str.cat has absorbing NA (good)
    assert obj_id.count() == len(obj_id)
    parsed['obj_id'] = obj_id

    return parsed

def csv_trim(df):
    COLUMNS = ['ID', 'box0', 'box1', 'box2', 'box3', 'box_num',
       'category', 'cloud_cover', 'country_code', 'full_path', 'fold', 'gsd',
       'img_filename', 'img_height', 'img_width', 'mean_pixel_height',
       'mean_pixel_width', 'multi_resolution_dbl', 'multi_resolution_end_dbl',
       'multi_resolution_max_dbl', 'multi_resolution_min_dbl',
       'multi_resolution_start_dbl', 'off_nadir_angle_dbl',
       'off_nadir_angle_end_dbl', 'off_nadir_angle_max_dbl',
       'off_nadir_angle_min_dbl', 'off_nadir_angle_start_dbl',
       'pan_resolution_dbl', 'pan_resolution_end_dbl',
       'pan_resolution_max_dbl', 'pan_resolution_min_dbl',
       'pan_resolution_start_dbl', 'scan_direction', 'sun_azimuth_dbl',
       'sun_azimuth_max_dbl', 'sun_azimuth_min_dbl', 'sun_elevation_dbl',
       'sun_elevation_max_dbl', 'sun_elevation_min_dbl', 'target_azimuth_dbl',
       'target_azimuth_end_dbl', 'target_azimuth_max_dbl',
       'target_azimuth_min_dbl', 'target_azimuth_start_dbl', 'timestamp',
       'utm', 'visible', 'wavelength_code']
    df = df[COLUMNS]
    assert not df.isnull().all().any()
    return df

def axis_centrality(x, w, W):
    c_naive = (x + 0.5*w) / W - .5
    rw = w/W
    max_naive_mag = .5 - .5*rw + 1e-99
    return c_naive / max_naive_mag * (1-rw)**2

def centrality(df, p=6):
    cx = axis_centrality(df.box0, df.box2, df.img_width)
    cy = axis_centrality(df.box1, df.box3, df.img_height)
    cr = (abs(cx)**p + abs(cy)**p) ** (1/p)
    return cr

def read_single_Plog(model_name):
    return pd.DataFrame.from_csv('working/single_model_prediction/%s.csv' % model_name)

def read_merged_Plog():
    return (
            lerp(.46,
                lerp(2/3,
                    lerp(0.55,
                        read_single_Plog('model01'),
                        lerp(0.5,
                            read_single_Plog('model02'),
                            read_single_Plog('model03'))
                        ),
                    lerp(0.5,
                        lerp(0.5, read_single_Plog('model04'),
                            read_single_Plog('model05')),
                        lerp(0.5, read_single_Plog('model06'),
                            read_single_Plog('model07'))
                        )
                    ),
                lerp(0.4,
                    lerp(0.5, 
                        read_single_Plog('model08'),
                        read_single_Plog('model09')),
                    lerp(2/3,
                        read_single_Plog('model10'),
                        lerp(0.5,
                            read_single_Plog('model11'),
                            read_single_Plog('model12'))
                        )
                    )
                )
            )

def softmax(Plog):
    P = np.exp(Plog)
    return P.div(P.sum(1), 0)

def create_submission(P):
    CATS = P.columns
    threshold = 0
    expected_qty = P.sum() + 1e-30
    for i in range(5):
        value = (P - threshold) / expected_qty * cat_impact.loc[CATS].fillna(0)
        p = value.apply(np.argmax, 1)
        pred_count = p.value_counts().loc[CATS].fillna(0)
        threshold = pd.Series(P.values[np.arange(len(P)), P.columns.get_indexer(p)], P.index).groupby(p).sum().loc[CATS].fillna(0) / (expected_qty + pred_count)
    return p
