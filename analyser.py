import nrlib
import cv2
import csv

function_mapper = {"blur":nrlib.calc_blur, "blur_ratio":nrlib.calc_blur_ratio, "noise":nrlib.calc_noise, "noise_ratio":nrlib.calc_noise_ratio, "blockiness":nrlib.calc_blockiness, "si":nrlib.calc_si, "ti":nrlib.calc_ti, "mi":nrlib.calc_mi}
PRINT_INTERVAL = 10
SAVE_INTERVAL = 50

def write_to_csv(dict, out="output.csv"):
    with open(out, "w") as outfile:
        writer = csv.writer(outfile, lineterminator = '\n')
        writer.writerow(dict.keys())
        writer.writerows(zip(*dict.values()))
        outfile.close()

def get_metrics():
    return function_mapper.keys()

def create_dict(m):
    dict = {}
    dict["frame"] = []
    for x in m:
        dict[x] = []
    return dict

def analyse_video(path, m, out="output.csv"):
    cap = cv2.VideoCapture(path)
    ret = True
    nr_dict = create_dict(m)
    pvs_lum = None
    f=1

    while(ret):
        ret, frame = cap.read()
        if(not ret):
            break
        lum = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:,:,0]

        for x in m:
            if(x in ["ti","mi"]):
                if(pvs_lum is None):
                    nr_dict[x].append(0.0)
                else:
                    nr_dict[x].append(function_mapper[x](pvs_lum, lum))
            else:
                nr_dict[x].append(function_mapper[x](lum))

        nr_dict["frame"].append(f)

        pvs_lum = lum
        if(f%PRINT_INTERVAL==0):
            print("Analysed "+str(f)+" frames")
        f+=1

        if(f%SAVE_INTERVAL==0):
            print("Saving...")
            write_to_csv(nr_dict, out)

    write_to_csv(nr_dict, out)
    print("Calculation completed")
