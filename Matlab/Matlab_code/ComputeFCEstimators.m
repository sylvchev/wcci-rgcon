%% script to compute features on Challence BCI dataset in an automatic fashion via Brainstorm
    % Author: MCC
    % Creation: 19/06/2020

% Go to /brainstorm3, launch brainstorm by writing "brainstorm" in the command window
    ProtocolName='RIGOLETTO';
    iProtocol=bst_get('Protocol',ProtocolName);
    gui_brainstorm('SetCurrentProtocol',iProtocol);
 
%% load filenames
load sFiles

%% FC estimation
   TimeWin=[3, 7.5]; % time window to consider
   DoMy_Conn_ICoh(sFiles,TimeWin); % estimation / frequency bin, need to average afterwards
   DoMy_Conn_Coh(sFiles,TimeWin); % estimation / frequency bin, need to average afterwards
   
   Freq_Band={'alpha-beta', '8, 30', 'mean'};
   DoMy_Conn_PLV(sFiles, TimeWin, Freq_Band);
   DoMy_Conn_AEC(sFiles, TimeWin, Freq_Band);

%% Data storage in (12 channels x 12 channels x 80 trials) matrices for a given subject (training set)
root='.../RIGOLETTO/data/';
root_db='.../Data'; % path where the .mat files from the challenge are stored

% Frequency bin index
idxStartAlpha=4; idxEndAlpha=6;
idxStartBeta=7; idxEndBeta=15;

nb_subj=8; % number of subjects
nb_chan=12; % number of channels
nb_Freq=1;
MatICoh=[];
MatCoh=[];
MatPLV=[];
MatAEC=[];

for i=1:nb_subj
       load(strcat(root_db,'parsed_P0',num2str(i),'T.mat'));

       % ICoh & Coh (ICoh computed first)
        cd(strcat(root,'parsed_P0',num2str(i),'T/parsed_P0',num2str(i),'T_all'));
        temp= dir('timefreq_connectn_cohere*');
        
        tempdata=[];
        for j=1:80
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,idxStartAlpha:idxEndBeta),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end
        MatICoh{i,1}=tempdata;
        MatICoh{i,2}=Labels;
         
        tempdata=[];
        for j=81:160
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,idxStartAlpha:idxEndBeta),3);
            tempdata(:,:,j-80)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end   
        MatCoh{i,1}=tempdata;
        MatCoh{i,2}=Labels;
        
        % AEC
        cd(strcat(root,'parsed_P0',num2str(i),'T/parsed_P0',num2str(i),'T_all'));
        temp= dir('timefreq_connectn_aec*');
        tempdata=[];
        for j=1:length(temp)
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,1:2),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end
        MatAEC{i,1}=tempdata;
        MatAEC{i,2}=Labels;
        
        % PLV
        cd(strcat(root,'parsed_P0',num2str(i),'T/parsed_P0',num2str(i),'T_all'));
        temp = dir('timefreq_connectn_plv*');
        for j=1:length(temp)
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,1:2),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end       
        MatPLV{i,1}=tempdata;
        MatPLV{i,2}=Labels;
        

end

save(strcat(root_db,'ICoh_Training_121280.mat'),'FreqInfos','MatICoh','Labels');
save(strcat(root_db,'Coh_Training_121280.mat'),'FreqInfos','MatCoh','Labels');
save(strcat(root_db,'AEC_Training_121280.mat'),'FreqInfos','MatAEC','Labels');
save(strcat(root_db,'PLV_Training_121280.mat'),'FreqInfos','MatPLV','Labels');

%% testing files
clc

MatICoh=[];
MatCoh=[];
MatPLV=[];
MatAEC=[];
for i=1:nb_subj
        cd(strcat(root,'parsed_P0',num2str(i),'E/parsed_P0',num2str(i),'E_all'));
        temp= dir('timefreq_connectn_cohere*');
        
        tempdata=[];
        for j=1:40
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,idxStartAlpha:idxEndBeta),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end
        MatICoh{i,1}=tempdata;
         
        tempdata=[];
        for j=41:80
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,idxStartAlpha:idxEndBeta),3);
            tempdata(:,:,j-40)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end   
        MatCoh{i,1}=tempdata;
        
        % AEC
        cd(strcat(root,'parsed_P0',num2str(i),'E/parsed_P0',num2str(i),'E_all'));
        temp= dir('timefreq_connectn_aec*');
        tempdata=[];
        for j=1:length(temp)
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,1:2),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end
        MatAEC{i,1}=tempdata;
        
        % PLV
        cd(strcat(root,'parsed_P0',num2str(i),'E/parsed_P0',num2str(i),'E_all'));
        temp = dir('timefreq_connectn_plv*');
        for j=1:length(temp)
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,1:2),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end       
        MatPLV{i,1}=tempdata;
             
end

save(strcat(root_db,'ICoh_Testing_121240.mat'),'FreqInfos','MatICoh');
save(strcat(root_db,'Coh_Testing_121240.mat'),'FreqInfos','MatCoh');
save(strcat(root_db,'AEC_Testing_121240.mat'),'FreqInfos','MatAEC');
save(strcat(root_db,'PLV_Testing_121240.mat'),'FreqInfos','MatPLV');

%% Compute covariance on band-pass filtered data (8-30Hz)
clc
% training file
nb_subj=8;
MatCov=[];

for i=1:nb_subj
       load(strcat(root_db,'parsed_P0',num2str(i),'T.mat'));
        cd(strcat(root,'parsed_P0',num2str(i),'T/parsed_P0',num2str(i),'T_all'));
        temp= dir('*_band.mat');
        tempCov=[];
        for j=1:length(temp)
            load(temp(j).name);
            tempCov(:,:,j)=cov(F');    
        end
        MatCov{i,1}=tempCov;
        MatCov{i,2}=Labels;
end
save(strcat(root_db,'Cov_Training_All.mat'),'MatCov');

% testing files
clc
MatCov=[];
nb_subj=8;
for i=1:nb_subj
        cd(strcat(root,'parsed_P0',num2str(i),'E/parsed_P0',num2str(i),'E_all'));
        % ICoh & Coh
        temp_all= dir('*_band.mat');
        tempCov=[];
        for j=1:length(temp_all)
            load(temp_all(j).name);
            tempCov(:,:,j)=cov(F'); 
        end
        MatCov{i,1}=tempCov;
end
save(strcat(root_db,'Cov_Testing.mat'),'MatCov');

%% Subject 09 & 10
clc

MatICoh=[];
MatCoh=[];
MatPLV=[];
MatAEC=[];
MatCov=[];

for i=9:10
    if i==9
        cd(strcat(root,'parsed_P0',num2str(i),'E/parsed_P0',num2str(i),'E_all'));
    else
        cd(strcat(root,'parsed_P',num2str(i),'E/parsed_P',num2str(i),'E_all'));
    end
        temp= dir('timefreq_connectn_cohere*');
        
        tempdata=[];
        for j=1:40
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,idxStartAlpha:idxEndBeta),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end
        MatICoh{i-8,1}=tempdata;
         
        tempdata=[];
        for j=41:80
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,idxStartAlpha:idxEndBeta),3);
            tempdata(:,:,j-40)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end   
        MatCoh{i-8,1}=tempdata;
        
        % AEC
        temp= dir('timefreq_connectn_aec*');
        tempdata=[];
        for j=1:length(temp)
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,1:2),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end
        MatAEC{i-8,1}=tempdata;
        
        % PLV
        temp = dir('timefreq_connectn_plv*');
        for j=1:length(temp)
            load(temp(j).name)
            FreqInfos=Freqs;
            vector=mean(TF(:,1,1:2),3);
            tempdata(:,:,j)=DoMyMatrixFromVect(vector,nb_chan,nb_Freq);
        end       
        MatPLV{i-8,1}=tempdata;

        
        % Cov from filtered data
        temp_all= dir('*_band.mat');
        tempCov=[];
        for j=1:length(temp_all)
            load(temp_all(j).name);
            tempCov(:,:,j)=cov(F'); 
        end
        MatCov{i-8,1}=tempCov;
end

save(strcat(root_db,'ICoh_Testing_P09P10_121240.mat'),'FreqInfos','MatICoh');
save(strcat(root_db,'Coh_Testing_P09P10_121240.mat'),'FreqInfos','MatCoh');
save(strcat(root_db,'AEC_Testing_P09P10_121240.mat'),'FreqInfos','MatAEC');
save(strcat(root_db,'PLV_Testing_P09P10_121240.mat'),'FreqInfos','MatPLV');
save(strcat(root_db,'Cov_Testing_P09P10.mat'),'MatCov');