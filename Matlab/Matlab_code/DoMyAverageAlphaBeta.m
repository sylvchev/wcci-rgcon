function AlphaBeta_output=DoMyAverageAlphaBeta(variable,idx_Alpha,idx_Beta)

    AlphaBeta_output=variable;
    AlphaBeta_output.TF=[];
    selection_alpha=variable.TF(:,1,idx_Alpha);
    selection_beta=variable.TF(:,1,idx_Beta);
    AlphaBeta_output.TF(:,1,1)=mean(selection_alpha,3);
    AlphaBeta_output.TF(:,1,2)=mean(selection_beta,3);
    AlphaBeta_output.Freqs=[variable.Freqs(idx_Alpha(1));variable.Freqs(idx_Beta(1))];


end