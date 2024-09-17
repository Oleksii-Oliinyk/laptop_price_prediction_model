y_column = ['Price_euros']


X_columns = ["Inches", 
             "Ram", 
             "Weight", 
             "ScreenW", 
             "ScreenH", 
             "CPU_freq", 
             "PrimaryStorage", 
             "SecondaryStorage", 
             "Company_freq_encoded", 
             "TypeName_freq_encoded", 
             "Screen_freq_encoded", 
             "Touchscreen_freq_encoded", 
             "IPSpanel_freq_encoded", 
             "RetinaDisplay_freq_encoded", 
             "CPU_company_freq_encoded", 
             "CPU_model_freq_encoded", 
             "PrimaryStorageType_freq_encoded",
             "SecondaryStorageType_freq_encoded", 
             "GPU_company_freq_encoded", 
             "GPU_model_freq_encoded"]

outlier_columns = ["Ram", 
                   "Weight", 
                   "PrimaryStorage"]

scaling_columns = ["Inches", 
                   "Ram", 
                   "Weight", 
                   "ScreenW", 
                   "ScreenH", 
                   "CPU_freq", 
                   "PrimaryStorage", 
                   "SecondaryStorage"]


cat_columns = ['Company',
               'TypeName', 
               'Screen', 
               'Touchscreen', 
               'IPSpanel', 
               'RetinaDisplay', 
               'CPU_company', 
               'CPU_model', 
               'PrimaryStorageType', 
               'SecondaryStorageType',
               'GPU_company', 
               'GPU_model',]



