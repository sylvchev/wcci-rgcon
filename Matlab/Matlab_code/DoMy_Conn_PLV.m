function DoMy_Conn_PLV(sFiles, TimeWin, Freq)

% Start a new report
bst_report('Start', sFiles);

% Process: Phase locking value NxN
sFiles = bst_process('CallProcess', 'process_plv1n', sFiles, [], ...
    'timewindow',   TimeWin, ...
    'dest_sensors', 'EEG', ...
    'includebad',   1, ...
    'freqbands',    Freq, ...
    'mirror',       0, ...
    'keeptime',     0, ...
    'plvmeasure',   2, ...  % Magnitude
    'outputmode',   1);  % Save individual results (one file per input file)

% Save and display report
ReportFile = bst_report('Save', sFiles);
bst_report('Open', ReportFile);
% bst_report('Export', ReportFile, ExportDir);



end