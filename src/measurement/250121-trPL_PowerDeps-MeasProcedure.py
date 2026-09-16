import sys
sys.path.append(r"C://PyDAQmx-1.4.6")
sys.path.append(r"C://Users//PL Measurator")
import time
import ni_pulser_loop_new as DAQpulser
from snAPI.Main import *
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
import pandas as pd
import os
from pylablib.devices import Thorlabs
import numpy as np


from ctypes import cdll,c_long, c_ulong, c_uint32,byref,create_string_buffer,c_bool,c_char_p,c_int,c_int16,c_double, sizeof, c_voidp
from TLPM import TLPM
def launch_trPL(sn: snAPI, time_s: int, save_directory: float, pm: TLPM, wh: Thorlabs.FW, syncEdge: bool = 1, syncLevel:int = -300, syncOffset:int = 0, chan1Edge:bool = 0, chan1Level:int = -120, chan1Offset:int = 0, resolution_s:float = 1e-9, chanStop:int = 5, autoND = False):
    #sn = snAPI()
    # get first available device
    sn.getDevice()
    sn.setLogLevel(logLevel=LogLevel.DataFile, onOff=True)
    #initialize the device
    sn.initDevice(MeasMode.Histogram)
    # set the configuration for your device type
    binning_factor = int(np.log2(resolution_s/(250*1e-12)))
    sn.device.setHistoLength(chanStop) #2^{(10 + \mathrm{lengthCode})}
    sn.device.setBinning(binning_factor) # binsize = 250ps * 2**n
    #sync
    sn.device.setSyncEdgeTrig(syncLevel, syncEdge)
    sn.device.setSyncChannelOffset(syncOffset)
    #input channel
    sn.device.setInputEdgeTrig(0,chan1Level,chan1Edge)
    sn.device.setInputChannelOffset(0, chan1Offset)
    cntRt = sn.getCountRates()
    print(cntRt)
    if(cntRt[1] > 8e6):
       print("Count Rates are too high, counts = ", cntRt)
    else:
        sn.histogram.measure(acqTime = int(1e3*time_s), savePTU=True, waitFinished=False)

    if not(autoND):
        pos = 5
        while(pos > 0):
            wh.set_position(pos)
            cntRt = sn.getCountRates()
            #if bigger than the 15%
            if (cntRt[1] > 0.15*cntRt[0]):
                wh.set_position(pos+1)
                print("Chosen ND"+str(pos+1))
                break
            elif ((pos == 1) and (cntRt[1] < 0.05*cntRt[0])):
                print("Chosen ND"+str(pos))
            pos = pos - 1
    else:
        if(autoND < 7 and autoND > 0):
            wh.set_position(autoND)
        else:
            print('ND filter error')
            return None

   
    while True:
        finished = sn.histogram.isFinished()
        data, bins = sn.histogram.getData()
        plt.figure(1)
        if len(data):
            plt.clf()
            #plt.semilogy(bins, data[0], linewidth=2.0, label='sync')
            for c in range(1, 1+sn.deviceConfig["NumChans"]):
                plt.semilogy(bins, data[c], linewidth=2.0, label=f'chan{c}', alpha = 0.5)
            plt.xlabel('Time [ps]')
            plt.ylabel('Counts')
            plt.xlim([0,40e6])
            plt.ylim([0.1,1e5])
            plt.legend()
            plt.title("Counts / Time")
            plt.pause(1)

        if finished:
            measurement = np.array([bins, data[1]])
            df = pd.DataFrame(data = measurement.transpose(), columns = ["bins [ps]", "counts [#]"])
            newname = list(os.path.splitext(os.path.basename(save_directory)))
            newname.append(newname[-1])
            #read the power from power meter.
            power =  c_double()
            pm.measPower(byref(power))
            cntRt = sn.getCountRates()
            #create add on string
            add_on = "_"+str(time_s)+"s_"+str(int(1e-3*cntRt[0]))+"kHz_ND"+str(wh.get_position())+"_"+str(int(cntRt[1]))+"cps"+"_"+str(1e6*power.value)+"uW"
            newname[1] = add_on
            a = ""
            for n in newname:
                a = str(a)+str(n)

            #Save file with measurement parameters
            save_directory = os.path.join(os.path.dirname(save_directory), a)
            df.to_csv(save_directory, sep="\t")
            break

    return

def launch_LPtrPL(sn: snAPI, p: DAQpulser, time_s: int, pulseLength_s: int, repRate_Hz: int, save_directory: float, syncEdge: bool = 0, syncLevel:int = -250, syncOffset_ps:int = -99999, chan1Edge:bool = 1, chan1Level:int = -120, chan1Offset_ps:int = 99999, resolution_s:float = 1e-9, chanStop:int = 5):
    #sn = snAPI()
    p.start_pulses(pulseLength_s, repRate_Hz)
    # get first available device
    sn.getDevice()
    sn.setLogLevel(logLevel=LogLevel.DataFile, onOff=True)
    #initialize the device
    sn.initDevice(MeasMode.Histogram)
    # set the configuration for your device type
    binning_factor = int(np.log2(resolution_s/(250*1e-12)))
    sn.device.setHistoLength(chanStop) #2^{(10 + \mathrm{lengthCode})}
    sn.device.setBinning(binning_factor) # binsize = 250ps * 2**n
    #sync
    sn.device.setSyncEdgeTrig(syncLevel, syncEdge)
    sn.device.setSyncChannelOffset(syncOffset_ps)
    #input channel
    sn.device.setInputEdgeTrig(0,chan1Level,chan1Edge)
    sn.device.setInputChannelOffset(0, chan1Offset_ps)
    cntRt = sn.getCountRates()
    if(cntRt[1] > 8e6):
       print("Count Rates are too high, ND wheel is changed = ", cntRt)
    else:
        sn.histogram.measure(acqTime = int(1e3*time_s),savePTU=True, waitFinished=False)

    while True:
        finished = sn.histogram.isFinished()
        data, bins = sn.histogram.getData()
        plt.figure(10)
        if len(data):
            plt.clf()
            #plt.semilogy(bins, data[0], linewidth=2.0, label='sync')
            for c in range(1, 1+sn.deviceConfig["NumChans"]):
                plt.semilogy(bins, data[c], linewidth=2.0, label=f'chan{c}', alpha = 0.5)
            plt.xlabel('Time [ps]')
            plt.ylabel('Counts')
            plt.xlim([0,5e6])
            plt.ylim([0.1,1e5])
            plt.legend()
            plt.title("Counts / Time")
            plt.pause(1)

        if finished:
            measurement = np.array([bins, data[1]])
            df = pd.DataFrame(data = measurement.transpose(), columns = ["bins [ps]", "counts [#]"])
            newname = list(os.path.splitext(os.path.basename(save_directory)))
            newname.append(newname[-1])
            #read the thorlabs filter wheel
            wheel = Thorlabs.FW("COM8")
            add_on = "_"+str(time_s)+"s_"+str(int(cntRt[0]))+"Hz_ND"+str(wheel.get_position())+"_PW-"+str(1e6*pulseLength_s)+"us_"+str(int(cntRt[1]))+"cps"
            wheel.close()
            newname[1] = add_on
            a = ""
            for n in newname:
                a = str(a)+str(n)

            #Save file with measurement parameters
            save_directory = os.path.join(os.path.dirname(save_directory), a)
            df.to_csv(save_directory, sep="\t")
            wheel.close()
            p.stop_tasks()
            p.clear_tasks()
            break

    return

def initialize_power_and_motor(pm_wavelength: float = 705.00):
    # Initialising the power meter.
    tlPM = TLPM()
    deviceCount = c_uint32()
    tlPM.findRsrc(byref(deviceCount))
    print("PM devices found: " + str(deviceCount.value))
    resourceName = create_string_buffer(1024)
    for i in range(0, deviceCount.value):
        tlPM.getRsrcName(c_int(i), resourceName)
        print(c_char_p(resourceName.raw).value)
        break
    tlPM.close()
    tlPM = TLPM()
    tlPM.open(resourceName, c_bool(True), c_bool(True))
    tlPM.setWavelength(c_double(pm_wavelength))

    # Initialising the motor
    LS150 = Thorlabs.KinesisMotor("45176994")
    #Swipe the linear stage
    alpha = 61440000/(150) #steps/mm
    #Home the motor
    LS150.setup_homing(home_direction=None, limit_switch=None, velocity=43980465, offset_distance=None, channel=None, scale=None)
    LS150.home(force=True, sync=False)
    while (LS150.is_homing()):
        print("im homing homie")
        time.sleep(2)

    return tlPM, LS150, alpha

def get_refpowertables(stepSize_mm, pm: TLPM, stage: Thorlabs.KinesisMotor, alpha_steppermm: float,zero_mm: float = 12, max_dist: float = 89):
    increment_mm = 0 

    distance = np.empty(0)
    power_measurements = np.empty(0)

    while(increment_mm < max_dist):
        print("Calibration: im moving to "+str(increment_mm)+" mm")
        #move motor
        stage.move_to((zero_mm+increment_mm)*alpha_steppermm)
        stage.wait_move()
        distance = np.append(distance, float(float(tlS.get_position())/alpha_steppermm) - zero_mm)
        time.sleep(2)
        #measure power 5 several times
        powers =  np.empty(0)
        for i in range(5):  
            power =  c_double()
            pm.measPower(byref(power))
            powers = np.append(powers, power.value)
            time.sleep(0.5)
        power_measurements = np.append(power_measurements, np.mean(powers))

        #increment_mm the motor distance
        increment_mm = increment_mm+stepSize_mm

    #print("im moving back to 0")
    #stage.move_to((zero_mm)*alpha_steppermm)
    #stage.wait_move()

    return distance, power_measurements

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def go_to_power(power_goal: float, stage: Thorlabs.KinesisMotor, distances: np.array, measured_powers: np.array, alpha_steppermm: float, zero_mm: float = 12):

    if((power_goal > np.amax(measured_powers)) or (power_goal < np.amin(measured_powers))):
        print("Desired power is outside attainable powers, we go to edge power")

    id = find_nearest(measured_powers, power_goal)
    stage.move_to((zero_mm+distances[id])*alpha_steppermm)
    stage.wait_move()

    return 0


if(__name__ == "__main__"):
    #Set Devices
    sn = snAPI()
    #Set power dependencies
    powers_10k = 0.02e-6*np.logspace(0, 1.7781512503836436, 4)
    powers_5k = 0.5*powers_10k
    #Set folder and file names
    folderpath = r"D:\\Maxim\\20260115_MostafaEPFL\\TwoStep"
    if (os.path.isdir(folderpath)):
        print("Folder Path Correct.")
    else:
        print("Folder Path Incorrect.")
        sys.exit(1)
    name = "-FAPIRef2.dat"

    tlPM, tlS, alpha = initialize_power_and_motor()
    stepsize = 2
    distances, powers_measured = get_refpowertables(stepsize, tlPM, tlS, alpha, zero_mm = 12)

    wheel = Thorlabs.FW("COM8")

    times = [3600,3600,2400,2400]
    for i in range(len(powers_10k)):
        time.sleep(2.0)
        go_to_power(powers_10k[i], tlS, distances, powers_measured, alpha)
        filepath = folderpath+r"\\M"+str(i)+name
        print(filepath)
        launch_trPL(sn, times[i], filepath, tlPM, wheel, resolution_s = 2000e-12)
        time.sleep(2.0)

    go_to_power(powers_10k[0], tlS, distances, powers_measured, alpha)
    filepath = folderpath+r"\\M"+str(i+1)+name
    print(filepath)
    launch_trPL(sn, 1800, filepath, tlPM, wheel, resolution_s = 2000e-12) #30 minutes back to power 0 to check for consistent measurement

    go_to_power(0.01e-6, tlS, distances, powers_measured, alpha)
    time.sleep(2.0)
    wheel.set_position(6)

    tlS.close()
    tlPM.close()
    wheel.close()