%% Load FIS
fis = readfis('FL_IndicatorCharging_Sugeno.fis');

fprintf('Charging Infrastructure readiness for evacuation Evaluation.');
fprintf(['Please rate each indicator on a scale from 0 (bad) to 5 (excellent)',...
    'according to the notation tables:\n\n']);

% Get user inputs
CI1 = input('CI1 - Mobile/Backup Charging Access (0-5): ');
% Community access to mobile charging/backup power to support EV 
% chargers at high risk of congestion or power loss
CI2 = input('CI2 - Energy Requirements Estimation (0-5): ');
%  Estimation of Energy Requirements for EV-Based Evacuation
CI3 = input('CI3 - Public EV Charging Infrastructure Readiness (0-5): ');
% Readiness of Public EV Charging Infrastructure 
CI4 = input('CI4 - Heavy Duty EV/ESV Support (0-5): ');
% Community has considered the charging of electric heavy-duty vehicles 
% which are either helping in charging (mobile charging units) or 
% directly in evacuation
CI5 = input('CI5 - Charging Management and Emergency Priority Access (0-5): ');
% Charging Management and Emergency Priority Access for Ev

% Validate inputs
% Inputs: CI1..CI5 are in [0,5]
inputs = [CI1 CI2 CI3 CI4 CI5];


% ------- 0) Bounds -------
if any(inputs < 0) || any(inputs > 5)
    error('All ratings must be between 0 and 5.');
end

% ------- 1) CI2 = foundation (Bad => stop) -------
if CI2 < 1.5   % [0,1.5)
    error(['You cannot pursue this evaluation because the main indicator (CI2) is ', ...
           'Bad (<1.5). Please estimate evacuation energy needs before moving forward.']);
end

% ------- 2) CI3 = public infrastructure (Bad => stop) -------
if CI3 < 1.5   % [0,1.5)
    error(['You cannot pursue this evaluation because CI3 (Public EV Charging Infrastructure) ', ...
           'is Bad (<1.5). Strengthen baseline public charging readiness first.']);
end

% ------- 3) Compensation rule consistent with the decision tree -------
% If CI3 is Medium [1.5,3), require CI1 >= 3 (Good/Excellent). 
if any((CI3 >= 1.5 & CI3 < 3) & (CI1 < 3))
    error(['With CI3 at Medium (1.5–<3), CI1 (Mobile/Backup Charging) must be at least Good (>=3) ', ...
           'to continue. Improve mobile/backup charging plans or protocols.']);
end

% If CI3 is Good/Excellent (>=3), proceed regardless of CI1.
% ---> evaluation can continue
fprintf('OK. Proceeding with fuzzy evaluation.\n');


% Evaluate FIS
result = evalfis(fis, inputs);

% Display output
fprintf(['\nYour overall Charging Infrastructure Readiness Indicator',...
    'Score is: %.2f (out of 5)\n'], result);

% New case for result <= 0
if result <= 0
    fprintf('Your readiness is VERY BAD. Immediate actions required:\n');
    fprintf('- Review all infrastructure categories, as the system indicates a critical lack in all areas.\n');
    fprintf('- Start with Energy Requirements Estimation (CI2) and Public EV Charging Infrastructure (CI3).\n');
    fprintf('- Prioritize acquiring backup charging systems (CI1) and secure mobile charging units.\n');
end

% Case for result <= 1.5 (BAD)
if result <= 1.5
    fprintf('Your overall readiness is BAD. Prioritize improving:\n');
    if CI2 <= 2
        fprintf('- Energy Requirements Estimation (CI2): This is critical. Revise your energy modeling.\n');
    end
    if CI3 <= 2
        fprintf('- Public EV Charging Infrastructure Readiness (CI3): Upgrade coverage and planning.\n');
    end
    if CI1 <= 2
        fprintf('- Mobile/Backup Charging Access (CI1): Develop deployment strategies.\n');
    end
end

% Case for 1.5 < result <= 3 (MEDIUM)
if result > 1.5 && result <= 3
    fprintf('Your readiness is MEDIUM. To reach GOOD, focus on:\n');
    if CI1 <= 3
        fprintf('- Mobile/Backup Charging Access (CI1): Enhance coverage and agreements.\n');
    end
    if CI3 <= 3
        fprintf('- Public EV Charging Infrastructure Readiness (CI3): Improve congestion management and coverage.\n');
    end
    if CI2 <= 3
        fprintf('- Energy Requirements Estimation (CI2): Model evacuation scenarios more fully.\n');
    end
    if CI4 <= 3
        fprintf('- Heavy Duty EV/ESV Support (CI4): Plan for surge demand.\n');
    end
    if CI5 <= 3
        fprintf('- Charging Management and Emergency Prioritization (CI5): Define clear protocols.\n');
    end
end

% Case for 3 < result <= 4 (GOOD)
 if result > 3 && result <= 4
    fprintf('Your readiness is GOOD. To achieve EXCELLENT:\n');
    
    % Vérifie CI1
    if CI1 < 4
        fprintf('- Mobile/Backup Charging Access (CI1): Expand mobile units or backup agreements to improve resilience.\n');
    end
    
    % Vérifie CI2
    if CI2 < 4
        fprintf('- Energy Requirements Estimation (CI2): Refine scenario-based modeling for better accuracy.\n');
    end
    
    % Vérifie CI3
    if CI3 < 4
        fprintf('- Public EV Charging Infrastructure Readiness (CI3): Strengthen coverage in critical zones and reduce bottlenecks.\n');
    end
    
    % Vérifie CI4
    if CI4 < 4
        fprintf('- Heavy Duty EV/ESV Support (CI4): Improve surge load planning and real-time coordination.\n');
    end
    
    % Vérifie CI5
    if CI5 < 4
        fprintf('- Charging Management and Emergency Prioritization (CI5): Establish stronger protocols and agreements.\n');
    end
 end
 
% Case for 4.0 < result <= 4.5 (Near EXCELLENT)
if result > 4 && result <= 4.5
    fprintf('You are very close to EXCELLENT! Focus on fine-tuning:\n');
    if CI4 < 4.5
        fprintf('- Heavy Duty EV/ESV Support (CI4): Focus on surge load coordination and reserve capacity.\n');
    end
    if CI5 < 4.5
        fprintf('- Charging Management and Emergency Prioritization (CI5): Finalize emergency access protocols.\n');
    end
end

% Case for result > 4.5 (EXCELLENT)
if result > 4.5
    fprintf('Excellent readiness! Maintain your level with ongoing planning and regular updates.\n');
end