import ssgetpy
import os

matrix_names = [
    "Chem97ZtZ", "Dubcova1", "Dubcova2", "Dubcova3", "G2_circuit", "Kuu", "LF10000", 
    "LFAT5000", "Muu", "Pres_Poisson", "aft01", "apache1", "bcsstk08", "bcsstk09", "bcsstk10", 
    "bcsstk11", "bcsstk12", "bcsstk13", "bcsstk14", "bcsstk15", "bcsstk16", "bcsstk17", "bcsstk18", 
    "bcsstk21", "bcsstk23", "bcsstk25", "bcsstk26", "bcsstk27", "bcsstk28", "bcsstk36", "bcsstk38", 
    "bcsstm08", "bcsstm09", "bcsstm11", "bcsstm12", "bcsstm21", "bcsstm23", "bcsstm24", "bcsstm25", 
    "bcsstm26", "bcsstm39", "bloweybq", "bodyy4", "bodyy5", "bodyy6", "bundle1", "cant", "cbuckle", 
    "cfd1", "crystm01", "crystm02", "crystm03", "ct20stif", "cvxbqp1", "denormal", "ex10", "ex10hs", 
    "ex13", "ex15", "ex3", "ex33", "finan512", "fv1", "fv2", "fv3", "gridgena", "gyro", "gyro_k", "gyro_m", 
    "jnlbrng1", "mhd3200b", "mhd4800b", "minsurfo", "msc01050", "msc01440", "msc04515", "msc10848", "msc23052", 
    "nasa1824", "nasa2146", "nasa2910", "nasa4704", "nasasrb", "nd3k", "obstclae", "olafu", "parabolic_fem", 
    "plat1919", "plbuckle", "qa8fm", "raefsky4", "s1rmq4m1", "s1rmt3m1", "s2rmq4m1", "s2rmt3m1", "s3rmq4m1", 
    "s3rmt3m1", "s3rmt3m3", "shallow_water1", "shallow_water2", "sts4098", "t2dah_e", "t2dal_e", "t3dl_e", 
    "ted_B", "ted_B_unscaled", "thermal1", "thermomech_TC", "thermomech_TK", "thermomech_dM", "torsion1", 
    "vanbody", "wathen100", "wathen120"
]

save_directory = "../../matrices"

downloaded_count = 0

for matrix_name in matrix_names:
    print(f"Searching for matrix: {matrix_name}")
    
    matrices = ssgetpy.search(name=matrix_name)
    
    if matrices:
        matrix_entry = matrices[0]
        print(f"Downloading and extracting matrix: {matrix_entry.name}")
        
        matrix_path, extracted_path = matrix_entry.download(destpath=save_directory, format='MM', extract=True)
        
        if matrix_path.endswith(".tar.gz"):
            os.remove(matrix_path)
            print(f"Removed the zipped file: {matrix_path}")
        
        downloaded_count += 1
    else:
        print(f"Matrix {matrix_name} not found in SuiteSparse collection.")

print(f"Total matrices downloaded and extracted: {downloaded_count}")
