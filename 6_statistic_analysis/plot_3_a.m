c3
dependent_1s = [92.81
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

dependent_2s = [95.49 
99.03 
99.58 
87.29 
98.61 
95.21 
97.50 
98.68 
93.61 
96.46 
85.35 
92.50 
98.13 
98.47 
98.61 
99.79 
];

dependent_5s = [96.20 
99.10 
99.70 
90.30 
99.70 
96.40 
97.60 
99.10 
96.20 
97.40 
89.20 
92.00 
99.00 
98.40 
98.80 
100.00 


];

dependent_10s = [96.52 
99.31 
99.66 
89.21 
100.00 
96.18 
97.91 
98.61 
96.87 
95.47 
87.14 
94.10 
99.30 
98.61 
98.26 
100.00 

];

independent_1s = [92.81
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

independent_2s = [95.49 
99.03 
99.58 
87.29 
98.61 
95.21 
97.50 
98.68 
93.61 
96.46 
85.35 
92.50 
98.13 
98.47 
98.61 
99.79 
];

independent_5s = [91.56 
98.72 
99.27 
75.38 
98.12 
87.64 
93.37 
94.72 
90.59 
95.63 
76.18 
86.32 
97.43 
96.63 
98.33 
99.93 

];

independent_10s = [91.56 
98.72 
99.27 
75.38 
98.12 
87.64 
93.37 
94.72 
90.59 
95.63 
76.18 
86.32 
97.43 
96.63 
98.33 
99.93 

];

[h1, p1, ci1, stats1] = ttest(dependent_1s, independent_1s);
[h2, p2, ci2, stats2] = ttest(dependent_2s, independent_2s);
[h3, p3, ci3, stats3] = ttest(dependent_5s, independent_5s);
[h4, p4, ci4, stats4] = ttest(dependent_10s, independent_10s);



data = [dependent_1s dependent_2s dependent_5s dependent_10s];
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
plot(mean_data, 'o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'Color', 'k')

% 绘制标准误差
aa_data = std(data);
sem_data = std(data)/sqrt(size(data,1));
errorbar(mean_data, sem_data, 'LineStyle', 'none', 'LineWidth', 1.5, 'Color', 'k')

% 设置横轴标签和标题
set(gca, 'XTick', 1:4)
set(gca, 'XTickLabel', {'1s', '2s', '5s', '10s'})
xlabel('Decision Window')
ylabel('Decoding accuracy(%)')
title('')


% Add label for SOTA

plot(0.95, 90.6, 'p', 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'LineWidth', 2)

hold on
xlabel = [1.05 2 3 4];
stanet = [90.1 91.4 92.6 93.9];
plot(xlabel, stanet, 'd', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'Color', 'k')
rgc = [79 82 84.5 86];
plot(rgc, '^', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', 'Color', 'k')


legend('','','','','','','','','','','','','','','','','DenseNet-3D','','XAnet (Pahuja et.al, 2023)', 'Stanet (Su et.al, 2022)','RGC (Geirnaert et.al, 2021)');