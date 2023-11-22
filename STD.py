import pandas as pd
import numpy as np
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
#plt.show()

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
# plt.show()

plt.figure("ac ppg")
plt.plot(indices, ac_ppg_data)
# plt.show()

ampl, __ = find_peaks(median_data, distance=int(0.34 * fs), width = 0.1*fs, prominence = 15000)
print (ampl)
plt.figure("find peak ppg")
plt.plot(indices, median_data)
for value in ampl:
    plt.plot(value, median_data[value], "r*")
# plt.show()


import numpy as np

def calculate_hr(ampl, window_size, fs):
    """
    Tính HR (Heart Rate) dựa trên dữ liệu đầu vào.

    Tham số:
        - data: Dữ liệu đầu vào (một mảng, Series hoặc DataFrame).
        - window_size: Kích thước cửa sổ trượt.
        - fs: Tần số lấy mẫu.

    Trả về:
        - Kết quả HR được tính toán.
    """


# Sử dụng hàm calculate_hr với dữ liệu và thông số tương ứng
window_size = 3
fs = 500  # Ví dụ: Tần số lấy mẫu là 100 (đơn vị Hz)


a = 0

def rolling_mean(ampl1, window_size1, fs1):
    """
    Thực hiện trượt cửa sổ trên dữ liệu và tính giá trị biểu thức ((60 * fs) / (peak sau trừ peak trước)).

    Tham số:
        - data: Dữ liệu đầu vào (một mảng, Series hoặc DataFrame).
        - window_size: Kích thước cửa sổ trượt.
        - fs: Tần số lấy mẫu.

    Trả về:
        - Kết quả của việc tính giá trị biểu thức trên từng cửa sổ.
    """
    num_windows = len(ampl1) - window_size1 + 1
    print((num_windows))
    hr_results = []

    for a in range(num_windows - 2):
        sum_hr = 0

        for i in range(window_size1 ):
            hr = (60 * fs1) / (ampl1[a+i+1] - ampl1[a+i])
            sum_hr += hr

        avg_hr = sum_hr / (window_size1 - 1)
        hr_results.append(avg_hr)

    return hr_results


# rolling_mean_data = []
rolling_mean_data = rolling_mean(ampl, window_size, fs)


# In kết quả
print(rolling_mean_data)
# rolling_mean_data_1 = rolling_mean_data.flatten()
import matplotlib.pyplot as plt

# Số mẫu thu được sau khi trượt
num_samples = len(rolling_mean_data)
indices_1 = [i for i in range(len(rolling_mean_data))]
# Vẽ biểu đồ
print(len(indices_1))
plt.plot(indices_1, rolling_mean_data)
plt.xlabel('Số mẫu thu được sau khi trượt')
plt.ylabel('Heart Rate')
plt.title('HR theo số mẫu')
plt.grid(True)

# Hiển thị biểu đồ
plt.show()