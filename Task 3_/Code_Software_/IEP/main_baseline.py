import layer_lib
import configs
import os
import shutil
import random
import pandas as pd  # Import pandas

_CONFIG_PATH = "/Users/shuangfeng/Documents/git/ZEVEvacuation/python_codes/layer_model/configs.py"



def main():
    output = layer_lib.baseline()
    
    # Convert output to DataFrame
    no_charging_od_route_df = pd.DataFrame.from_dict(output.no_charging_od_route, orient='index')
    charging_od_route_df = pd.DataFrame.from_dict(output.charging_od_route,orient='index')
    
    # Write DataFrame to Excel
    result_dir=os.path.join(configs.BASE_DIR, 'baseline')
    os.makedirs(result_dir, exist_ok=True)
    output_file = os.path.join(result_dir, 'output.xlsx')
    print(result_dir)
    
    
if __name__ == "__main__":
    main()
