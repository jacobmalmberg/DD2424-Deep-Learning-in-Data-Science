function make_plot(x_axis, val_vec, x_label, y_label, name_array)
% to make plots like fig3/4
figure;
plot(x_axis, val_vec);
legend('validation');
ylabel(y_label); 
xlabel(x_label);
path = "Result_Pics/";
name = name_array;
title(name)
saveas(gcf, path + name +".png") %gcf = current
end