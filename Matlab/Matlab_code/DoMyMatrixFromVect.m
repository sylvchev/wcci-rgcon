function temp_out=DoMyMatrixFromVect(vector,nb_ROIs,nb_Freq)
for j=1:nb_Freq % 
    for k=1:nb_ROIs
        temp = vector(:,1,j);
        a = triu(ones(nb_ROIs));
        a(a > 0) = temp;
        temp_out = (a + a')./(eye(nb_ROIs)+1);        
    end
end
end
