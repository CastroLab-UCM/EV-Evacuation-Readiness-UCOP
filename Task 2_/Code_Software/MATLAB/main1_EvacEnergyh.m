%------------------------------------------
% Description: 
%       Evaluate the Total Charging distance 
%------------------------------------------
clc; close all; clear all;

%---------------------------------------------
%1. Define Evaluation Use Case
%---------------------------------------------
% initial SOC
%setup.prob.EV_soc.mode=0.1; % pessimistic SoC0
setup.prob.EV_soc.mode=0.5; %average SOC
%setup.prob.EV_soc.mode=0.9; %optmistic SoC0

% Distance to shelter
%distance.mode=[20: 20: 200];      % [km] distance to shelter
%distance.mode=[20, 50, 100, 150]; % [km] distance to shelter
distance.mode=150;

setup.N_vehicles = 1000; % number of vehicles used in the evacuation

setup.plot_on = 0; % plot signals?

%------------------------------------------------------
%2. Apply Monte Carlo Simulation to evaluate
%   Total Energy Needed for EV evacuation    
%------------------------------------------------------

% 3. Number of samples
%Nsamples = 500; % number of samples per distance
Nsamples = 10; % number of samples per distance
for j=1:length(distance.mode)
    for k=1:Nsamples
        setup.distance.mode=distance.mode(j);
        [e_ch_fit,E_ch_total]=EvacEnergy(setup);
        batch.EchTotal_instance(j,k)=E_ch_total;
    end
    batch.EchTotal_mean(j)=mean(batch.EchTotal_instance(j,:));
    batch.EchTotal_std(j)=std(batch.EchTotal_instance(j,:));
end

%% ------------------------------------
% Plot results
%-----------------------------------------
figure;

assert(setup.N_vehicles==1000); % plot assume that we simulated 1000 vehicles.

%plot standard variation of total Charging energy (as a confidence band)
x=distance.mode;
y=batch.EchTotal_mean/1000;
xconf = [x x(end:-1:1)] ;         
yconf = [y+batch.EchTotal_std/1000 y(end:-1:1)-batch.EchTotal_std(end:-1:1)/1000];
p = fill(xconf,yconf,'red');
p.FaceColor = [1 0.8 0.8];      
p.EdgeColor = 'none';       
hold on;

%plot  mean of total charging energy
plot(distance.mode, batch.EchTotal_mean/1000);
ylabel('Charging Energy per 1000 EVs [MWh]'); hold on
grid on; 
xlabel('distance [km]')

%% ------------------------------------
%plot histogram of total charging energy
figure
nbins = 20;
histfit(batch.EchTotal_instance(end,:)/1000,nbins,'normal');
grid on
xlabel('Total Charging Energy [MWh]')