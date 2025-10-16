function Charging_Analysis()
close all; clc; 
% --------- PARAMETERS ------------
setup.n_L2 = 1:1:2500; % number of L2 chargers
setup.p_L2_nom = 6;    % [kW] Level 2 charger power 
setup.p_L2_sigma = 3;  % [kW] Level 2 charger power 
setup.p_DC_nom = 80;   % [kW] DC fast charger power 
setup.p_DC_sigma = 20; % [kW] Level 2 charger power 
setup.beta = 0.98;     % [0-1] probability of delivering charging power
T_eva    = 4;          % [h] evacuation time

setup.plot_avg = 1;    % plot number of L2/DC chargers assuming average charging power? 
setup.plot_changeConstraint = 0;  % plot number of L2/DC charging assuming probabilistic model

setup.eta = 1-setup.beta;

figure; 
%-----------------------------------------------
% Number of L2/DC chargers available in California
% Data source: 
%   https://www.energy.ca.gov/data-reports/energy-almanac/zero-emission-vehicle-and-infrastructure-statistics-collection/electric
%----------------------------------------------------------------------------------------
%  couty name, numer of L2 chargers, number of DC chargers, marker
CEC_Data = { {'Mariposa',  32, 30, 'x'}, ...
             {'Merced',   102, 68, 'o'}, ...             
             {'Santa Cruz', 77, 282, '+'}, ...
             {'Fresno', 959, 309, "square"}, ...
             {'Sacramento', 1850, 320, "diamond"}, ...
             {'San Francisco', 2207, 209, "hexagram"}
             };

ix =1;
for j=1:length(CEC_Data)
  county_data = CEC_Data{j};
  example.l2 = county_data{2}; % number of L2 chargers 
  example.dc = county_data{3}; % number of DC chargers
  loglog(example.l2, example.dc, 'Marker',county_data{4},'MarkerSize',10);
  hold on;
  setup.legend{ix}=sprintf(county_data{1}); ix = ix+1;    
end
legend(setup.legend,'AutoUpdate','off');


%----------------------------------------------------------
%   Plot charging power for different number of EVs
%----------------------------------------------------------
%E_ch_ref_vect = [1, 5, 10, 25, 50, 100, 150]*1000; % [kWh] Total charging energy for all EVs
p_ch_ref_vect = [1, 2, 3, 6,  12, 18, 24,  30,  36, 50   ]; % [MW] Total charging power
E_ch_ref_vect = p_ch_ref_vect*T_eva*1000;

for j=1:length(E_ch_ref_vect)
    E_ch_ref = E_ch_ref_vect(j); %1*1e3;
    setup=plot_Energy(E_ch_ref, T_eva, setup);
end

%%
E_ch_ref = 10*1e3; % [kWh] charging power for all cars
setup.p_ch_ref = E_ch_ref/T_eva; % [kW] charging power

figure
plot_probabilistic_constraint(setup,setup.eta);

end

function setup=plot_Energy(E_ch_ref, T_eva, setup)

%E_ch_ref = 150*1e3; % [kWh] charging power for all cars
setup.p_ch_ref = E_ch_ref/T_eva; % [kW] charging power

% legend
if ~isfield(setup,'legend') % is this the first plot? 
    ix=0;
else
    ix=length(setup.legend);    
end
%setup.legend{ix+1}=sprintf('%d MWh/%dh', E_ch_ref/1000, T_eva); 
setup.legend{ix+1}=sprintf('%.0f MW', round(E_ch_ref/T_eva)/1000); 
%setup.legend{ix+1}=''; 
if setup.plot_avg 
    plot_bondary_ideal(setup, setup.legend{ix+1});
end
if setup.plot_changeConstraint
    plot_probabilistic_constraint(setup,setup.eta);
end

end

function plot_probabilistic_constraint(setup,eta)    
% range for analysis
    n_L2_range = [1:1:2500];
    n_dc_range = [1:1:3000];
    [L2, DC]=meshgrid(n_L2_range, n_dc_range);
    norminv(eta,0,1)
    
    p_ch_range = DC.*setup.p_DC_nom  +  ...
                    L2.* setup.p_L2_nom + ...
                     sqrt(DC.*(setup.p_DC_sigma.^2) + L2.*(setup.p_L2_sigma.^2)).*norminv(eta,0,1);
    
    ix=find(p_ch_range-setup.p_ch_ref>0);
    dcans=DC(ix);
    l2ans=L2(ix);
    %k=3
    %dcans(k)
    %l2ans(k)
    POINTS=[dcans,l2ans];
    [k,av] = convhull(POINTS);    
    %loglog(l2ans,dcans,'.','linestyle','none'); hold on
    

    hold on;
    grid on; 
    
end

function plot_bondary_ideal(setup, legend)

XMIN=10;
setup.n_dc = n_dc_est(setup.p_ch_ref, setup.n_L2, setup.p_L2_nom, setup.p_DC_nom );
loglog(setup.n_L2, setup.n_dc); hold on
text(XMIN,setup.n_dc(1), legend)

grid on;
xlabel('n_L_2 = number of L2 chargers [-]');
ylabel('n_D_C = number of DC chargers [-]');
%ylim([0 70])
ylim([XMIN, 1000])
xlim([XMIN, setup.n_L2(end)])
end

function n_dc=n_dc_est(p_ch_ref, n_L2, p_L2_nom, p_DC_nom )
%round
n_dc = ((p_ch_ref-n_L2*p_L2_nom)/p_DC_nom); % number of DC chargers
end