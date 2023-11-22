import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
with open('ppg_pcg//ppg_pcg_/93 57 56 dang.txt', 'r') as file:
    # Đọc nội dung từ tệp
    content = file.read()

# Loại bỏ tất cả các dấu cách và thay thế chúng bằng dấu phẩy
content_without_spaces = content.replace("\t", ",")

# In nội dung sau khi xử lý
#print(content_without_spaces)

# Tạo hoặc mở tệp mới để lưu nội dung đã xử lý
with open('PPG_PCG_file.csv', 'w') as new_file:
    # Ghi nội dung đã xử lý vào tệp mới
    new_file.write(content_without_spaces)


file_name = 'PPG_PCG_file.csv'  # Thay 'your_csv_file.csv' bằng tên thực tế của tệp

import csv
# Mở tệp CSV để đọc
ppg_data = []
pcg_data = []
fs = 500
windowsize = int(fs* 0.1)
with open(file_name, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Duyệt qua từng dòng trong tệp CSV
    for row in csv_reader:
        if len(row) >= 9:  # Đảm bảo có ít nhất 7 cột trong mỗi dòng
            column_7_data = float(row[7])
            column_6_data = float(row[6])# Lấy dữ liệu từ cột thứ 7 (0-based index)
            ppg_data.append(column_7_data)
            pcg_data.append(column_6_data)
            # Sử dụng dữ liệu từ cột thứ 7 ở đây, ví dụ:
            #print(column_data)
import matplotlib.pyplot as plt

# Dữ liệu mảng (ví dụ: danh sách các số nguyên từ 1 đến 10)
#data = [1, 2, 4, 6, 9, 11, 13, 14, 15, 16]
indices = [i for i in range(len(ppg_data))]
# plt.figure(9003)
# plt.plot(indices,ppg_data)
# plt.show()
ch = 1
fig, axs = plt.subplots(2, 1, sharex=True)
# fig.figure("ppg-pcg")
# for i in range(1, ch + 2):
#     if i != (ch + 1):
# plt.subplot(ch + 1, 1, i)
axs[0].plot(indices, ppg_data)
axs[0].set_xlabel("So mau")
axs[0].set_ylabel("Gia tri ppg raw")
axs[0].set_title("ppg data")
# axs[0].plot(indices, red_movmean_data_flatten)
# for value in ampl:
#     axs[0].plot(value, median_data[value],"r*")
        #for value in ampl:
        #plt.plot(value, median_data[value], "x")
    # else:
        # plt.subplot(ch + 1, 1, i)
        # plt.plot(indices, pcg_data)
axs[1].plot(indices, pcg_data)
axs[1].set_xlabel("So mau")
axs[1].set_ylabel("Gia tri pcg raw")
axs[1].set_title("pcg data")
# axs[1].plot(indices, ir_movmean_data_flatten)
# for value in ampl_pcg_data_filtered:
#     axs[1].plot(value, pcg_filtered[value], "r*")
        #plt.ylim([30, 120])
plt.show()

def movmean_data(A, k):
    x = A.rolling(k,min_periods= 1, center= True).mean().to_numpy()
    return x
def movmedian_data(A, k):
    x = A.rolling(k, min_periods= 1, center= True).median().to_numpy()
    return x
ppg_data_copy = ppg_data.copy()
ppg_data_copy_frame = pd.DataFrame(ppg_data_copy)
#ArrayRed1 = .copy()
#ArrayRed2 = pd.DataFrame()
median_data = movmedian_data(ppg_data_copy_frame, windowsize)
#median_data = median_data_frame.flatten()
median_data_frame = pd.DataFrame(median_data)
baseline_data_frame = movmean_data(median_data_frame, fs)
baseline_data = baseline_data_frame.flatten()
median_data = median_data.flatten()
ac_ppg_data = median_data - baseline_data

plt.figure("ppg and baseline")
plt.plot(indices, median_data)
plt.plot(indices, baseline_data)
plt.show()

plt.figure("ac ppg")
plt.plot(indices, ac_ppg_data)
plt.show()

ampl, __ = find_peaks(median_data, distance=int(0.34 * fs), width = 0.1*fs, prominence = 15000)

plt.figure("find peak ppg")
plt.plot(indices, median_data)
for value in ampl:
    plt.plot(value, median_data[value], "r*")
plt.show()

pcg_data_frame = pd.DataFrame(pcg_data)
pcg_median_data_frame = movmedian_data(pcg_data_frame, windowsize)
pcg_median_data = pcg_median_data_frame.flatten()
ampl1, __= find_peaks(pcg_median_data, distance=int(0.15 * fs), height=(300000,5000000) )

# find peak pcg data
import heartpy as hp
pcg_filtered = hp.filter_signal(pcg_data, cutoff = [25, 120], sample_rate = fs,order = 4, filtertype='bandpass')
indices_pcg_data_filtered = [i for i in range(len(pcg_filtered))]
#pcg_filtered = -pcg_filtered
ampl_pcg_data_filtered, __= find_peaks(pcg_filtered, distance=int(0.2 * fs),prominence = 300000)
indices_ampl_pcg_data_filtered = [i for i in range(len(ampl_pcg_data_filtered))]

ppg_filtered = hp.filter_signal(ppg_data, cutoff = [2, 200], sample_rate = fs,order = 4, filtertype='bandpass')
indices_ppg_data_filtered = [i for i in range(len(ppg_filtered))]
ampl_ppg_data_filtered, __= find_peaks(ppg_filtered, distance = int(0.34 * fs))
indices_ampl_ppg_data_filtered = [i for i in range(len(ampl_ppg_data_filtered))]
# plt.figure("find peak pcg")
# for i in range(1, ch + 2):
#     if i != (ch + 1):
#         plt.subplot(ch + 1, 1, i)
#         plt.plot(indices_pcg_data_filtered, pcg_data)
#         plt.xlabel('Số mẫu')
#         plt.ylabel('Gia tri ADC')
#         plt.title('PCG data raw')
#     else:
#         plt.subplot(ch + 1, 1, i)
#         plt.plot(indices_pcg_data_filtered, pcg_filtered)
#         plt.xlabel('Số mẫu')
#         plt.ylabel('Gia tri ADC')
#         plt.title('find peak PCG')
#         for value in ampl_pcg_data_filtered:
#             plt.plot(value, pcg_filtered[value], "r*")
# plt.show()

##############################################################
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(indices_pcg_data_filtered, pcg_data)
axs[0].set_xlabel("so mau")
axs[0].set_ylabel("gia tri adc pcg")
axs[0].set_title("PCG raw")

axs[1].plot(indices_pcg_data_filtered, pcg_filtered)
axs[1].set_xlabel("so mau")
axs[1].set_ylabel("gia tri adc pcg")
axs[1].set_title("PCG after filter")

plt.show()
##################################################################
#tính đường trung bình bằng tổng chia sổ phần tử
#cách vẽ đường bao ở đây

indices_ampl = [i for i in range(len(ampl))]
indices_ampl1 = [i for i in range(len(ampl_pcg_data_filtered))]
# plt.figure("ppg-pcg find peak")
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(indices, median_data)
axs[0].set_xlabel("so mau")
axs[0].set_ylabel("gia tri adc ppg")
axs[0].set_title(f'PPG after filter, sum peak = {len(ampl)}')

peaksY = [] #lưu lại các giá trị cao nhất
for value in ampl:
   axs[0].plot(value, median_data[value],"r*")
   peaksY.append(median_data[value])
print(f'peaksY = {peaksY}')
axs[0].plot(ampl, peaksY)

axs[1].plot(indices_pcg_data_filtered, pcg_filtered)
axs[1].set_xlabel("so mau")
axs[1].set_ylabel("gia tri adc pcg")
axs[1].set_title(f'PCG after filter, sum peak = {len(indices_ampl_pcg_data_filtered)}')


peaksY1 = [] #lưu lại các giá trị cao nhất
for value in ampl_pcg_data_filtered:
    axs[1].plot(value, pcg_filtered[value], "r*")
    peaksY1.append(pcg_filtered[value])
print(f'peaksY1 = {peaksY1}')
        #plt.ylim([30, 120])

axs[1].plot(ampl_pcg_data_filtered, peaksY1)

# Tính tổng từ các đỉnh
sum_ppg = sum(peaksY)  # Tổng PPG
sum_pcg = sum(peaksY1)  # Tổng PCG

# Tính giá trị trung bình bằng cách lấy tổng chia cho số đỉnh
avg_ppg = sum_ppg / len(peaksY)  # Đỉnh PPG
avg_pcg = sum_pcg / len(peaksY1)  # Đỉnh PCG

# Tạo một mảng có số phần tử bằng số phần tử của đường bao
avg_line_ppg = np.full(len(ampl), avg_ppg)  # Đường trung bình PPG
avg_line_pcg = np.full(len(ampl_pcg_data_filtered), avg_pcg)  # Đường trung bình PCG

# Vẽ đường trung bình
axs[0].plot(ampl, avg_line_ppg, 'g--', label='Đường trung bình PPG')
axs[1].plot(ampl_pcg_data_filtered, avg_line_pcg, 'g--', label='Đường trung bình PCG')


plt.show()

###################################################################################################

threshold = avg_ppg + 43*(10**4)
threshold1 = avg_pcg + 2.8*(10**5)
PPG_remove_peak = []
PCG_remove_peak = []
avg = int(len(ppg_data)/2)
print(avg)

for value in ampl:
    if median_data[value] > threshold:
        PPG_remove_peak.append(value)

if len(PPG_remove_peak) == 0:
    ppg_indices_left = avg
    ppg_indices_right = avg
else:
    ppg_indices_left = 0
    for value2 in PPG_remove_peak:
        if (value2 < avg):
            ppg_indices_left = value2
        else:
            ppg_indices_left = 0

    ppg_indices_right = 2
    print(f'PPG_remove_peak = {PPG_remove_peak}')
    for valuer in PPG_remove_peak:
        if valuer > int(avg):
            ppg_indices_right = valuer
        else:
            ppg_indices_right = len(median_data)
    # for value2 in PPG_remove_peak:
    #     if (value2 < avg):
    #         ppg_indices_left = value2
    #         ppg_indices_right = len(median_data)
    #     else:
    #         ppg_indices_left = 0
    #         ppg_indices_right = value2
    #     print(f'PPG_remove_peak = {PPG_remove_peak}')



for value in ampl_pcg_data_filtered:
    if pcg_filtered[value] > threshold1:
        PCG_remove_peak.append(value)
print(f'ampl1 = {ampl1}')
print(f'threshold1 = {threshold1}')
print(PCG_remove_peak)

# ppg_indices_left = 0
# for value2 in PPG_remove_peak:
#     if (value2 < avg):
#         ppg_indices_left = value2
#     else:
#         ppg_indices_left = 0

# ppg_indices_right = 2
# print(f'PPG_remove_peak = {PPG_remove_peak}')
# for valuer in PPG_remove_peak:
#     if valuer > int(avg):
#         ppg_indices_right = valuer
#     else:
#         ppg_indices_right = len(median_data)

pcg_indices_left = 0
for value1 in PCG_remove_peak:
    if (value1 < avg):
        pcg_indices_left = value1
indices_left = 0
if ppg_indices_left < pcg_indices_left:
    indices_left = pcg_indices_left
else:
    indices_left = ppg_indices_left

pcg_indices_right = 1
for valuer1 in PCG_remove_peak:
    if valuer1 > int(avg):
        pcg_indices_right = valuer1
    else:
        pcg_indices_right = len(pcg_filtered)
indices_right = 0
if ppg_indices_right < pcg_indices_right:
    indices_right = ppg_indices_right
else:
    indices_right = pcg_indices_right

print(f'ppg_indices_right = {ppg_indices_right}')
print(f'ppg_indices_left = {ppg_indices_left}')
print(f'pcg_indices_right = {pcg_indices_right}')
print(f'pcg_indices_left = {pcg_indices_right}')
print(f'indices_right = {indices_right}')
print(f'indices_left = {indices_left}')

PPG_data_after_cut = median_data[int (indices_left + 4.2*(10**2)) : int (indices_right - 4*(10**2))]
PCG_data_after_cut = pcg_filtered[int (indices_left + 4.2*(10**2)) : int ( indices_right - 4*(10**2))]

indices_cut = [i for i in range(len(PPG_data_after_cut))]
indices_cut1 = [i for i in range(len(PCG_data_after_cut))]
print(PPG_data_after_cut)
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(indices_cut, PPG_data_after_cut)
axs[0].set_xlabel("so mau")
axs[0].set_ylabel("gia tri adc ppg")
axs[0].set_title(f'PPG after cut')

axs[1].plot(indices_cut1, PCG_data_after_cut)
axs[1].set_xlabel("so mau")
axs[1].set_ylabel("gia tri adc pcg")
axs[1].set_title(f'PCG after cut ')

plt.show()

##########################################################
VTT = []

print(len(ampl))
print(len(ampl_pcg_data_filtered))

#for i in range(indices):
for value in indices_ampl:
    VTT.append([ (ampl[int(value)])] - (ampl_pcg_data_filtered[2*value]))
    # print(ampl[int(value) ])
    # print(int(ampl1[2*value ]))
indicesVTT = [i for i in range(len(VTT))]
averageVTT = sum(VTT) / len(VTT)
plt.figure(" VTT ")
plt.plot(indicesVTT,VTT)
plt.xlabel('Số mẫu')
plt.ylabel('Thời gian VTT')
plt.title(f'Biểu đồ VTT, Giá trị VTT trung bình: {averageVTT/fs}s' )
#plt.text(len(VTT), max(VTT), f'Trung bình: {averageVTT}', fontsize=12, ha='center')
#plt.text(0.1, 0.9, f'Trung bình: {averageVTT}', transform=plt.gcf().transFigure, fontsize=12)
plt.show()
ET = []
for value in indices_ampl_pcg_data_filtered:
    if(value % 2 == 0):
        ET.append(ampl_pcg_data_filtered[value+ 1] - ampl_pcg_data_filtered[value ])
print(ET)
averageET = sum(ET) / len(ET)
indicesET = [i for i in range(len(ET))]
plt.figure(" ET ")
plt.plot(indicesET,ET)
plt.xlabel('Số mẫu')
plt.ylabel('Thời gian ET')
plt.title(f'Biểu đồ ET, Giá trị ET trung bình: {averageET/fs}s' )
#plt.text(len(VTT), max(VTT), f'Trung bình: {averageVTT}', fontsize=12, ha='center')
#plt.text(0.1, 0.9, f'Trung bình: {averageVTT}', transform=plt.gcf().transFigure, fontsize=12)
plt.show()

###############################################################################
