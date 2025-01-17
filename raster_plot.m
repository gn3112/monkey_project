function raster_plot(trial,times)
for n = 1:1
    neuron_units = size(trial.spikes,1);
    t_all = size(trial.spikes,2);
    spikes_bin = trial.spikes;
    figure(1)
    hold on
    for t =times
        [idx,~] = find(spikes_bin(:,t) == 1);
        if idx ~= 0
            color = {'b','r','k','c','y'};
            t_plot(1:length(idx),:) = t;
            plot(t_plot,idx,'.','Color',color{n})
            hold on
            clearvars t_plot
        end
    end
end

figure(1)
hold on
xlabel('Time in ms')
ylabel('Neurons units')
plot([300 300],[1 100],'Color','r','LineWidth',1.5)
hold on
plot([t_all-100 t_all-100],[1 100],'Color','r','LineWidth',1.5)
end