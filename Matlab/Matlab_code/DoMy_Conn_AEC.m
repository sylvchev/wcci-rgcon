function DoMy_Conn_AEC(sFiles, TimeWin, Freq)

% Start a new report
bst_report('Start', sFiles);

% Process: Amplitude Envelope Correlation NxN
sFiles = bst_process('CallProcess', 'process_aec1n', sFiles, [], ...
    'timewindow',   TimeWin, ...
    'dest_sensors', 'EEG', ...
    'includebad',   1, ...
    'freqbands',    Freq, ...
    'isorth',       1, ...
    'outputmode',   1);  % Save individual results (one file per input file)

% Save and display report
ReportFile = bst_report('Save', sFiles);
bst_report('Open', ReportFile);
% bst_report('Export', ReportFile, ExportDir);



end