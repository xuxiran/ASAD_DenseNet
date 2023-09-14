clear
CNN_baseline = [79.06 
97.74 
96.15 
68.02 
95.63 
82.88 
88.06 
88.99 
79.31 
88.26 
67.33 
65.97 
93.47 
87.29 
82.26 
96.35 
];

DenseNet_37_I3D = [92.81
98.89
99.31
83.06
98.61
94.24
96.04
98.06
90.24
94.97
80.66
89.97
97.6
97.08
98.37
99.72
];

CNN_3D = [76.35 
98.58 
98.72 
73.13 
95.63 
85.03 
90.31 
89.65 
86.32 
90.49 
72.36 
74.13 
96.15 
96.28 
96.94 
99.76 



];

DenseNet_37_3D = [89.93
98.06
98.89
78.89
97.29
90.9
94.48
95.52
87.15
93.75
78.99
86.6
96.22
96.74
96.98
99.55
];

[h1, p1, ci1, stats1] = ttest(CNN_3D,CNN_baseline);
[h2, p2, ci2, stats2] = ttest(DenseNet_37_3D,CNN_3D);
[h3, p3, ci3, stats3] = ttest(DenseNet_37_I3D,DenseNet_37_3D);

data = [CNN_baseline CNN_3D DenseNet_37_3D DenseNet_37_I3D];
mean_data = mean(data, 1);

% 绘制每个被试的表现
set(groot, 'defaultAxesFontName','Times New Roman');
set(groot, 'defaultTextFontName','Times New Roman');
set(groot, 'defaultAxesFontSize', 14);
set(groot, 'defaultTextFontSize', 14);

figure
xlim([0.5, 4.5])
ylim([64, 104])
set(gca, 'FontWeight', 'bold');
hold on
box on;
load("col.mat")
% 绘制每个被试的表现并设置线的透明度为0.6和虚线样式
for i = 1:size(data, 1)
    line_color = col(:,i)';
    plot(data(i, :), 'Color', line_color, 'LineStyle', '--', 'LineWidth', 1.5, 'Marker', 'o', 'MarkerSize', 3, 'MarkerFaceColor', line_color, 'Color', [line_color,0.6])
end

% 绘制每种条件下的均值
plot(mean_data, '-o', 'LineWidth', 1.8, 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'Color', 'k')

% 绘制标准误差
sem_data = std(data)/sqrt(size(data,1));
errorbar(mean_data, sem_data, 'LineStyle', 'none', 'LineWidth', 1.5, 'Color', 'k')


% 设置横轴标签和标题
set(gca, 'XTick', 1:4)
set(gca, 'XTickLabel', {'CNN-baseline','CNN-3D', 'DenseNet-3D', 'DenseNet-3D'})
xlabel('Model')
ylabel('Decoding accuracy(%)')
title('')

