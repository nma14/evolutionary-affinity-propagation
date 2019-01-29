function idx = evolutionary_affinity_propagation(S, data,lam, gamma, omega, maxiter, conviter,burnin,metric,snMinSize,sniter,sigma2)
%     Run evolutionary affinity propagation (EAP) algorithm for clustering
%     data across time steps. 
%     Inputs:
%     S: cell array with similarity matrix for each time point. 
%         length(S)=T, for T the number of time steps.
%         S{t} will be a NxN array, for data containing N data points. 
%         If data point does not exist at time t, replace with NaN
%     data: cell array with data points. 
%         length(data)=T
%         data{t} will be Nxd for d-dimensional data.
%     lam: message damping parameter
%     gamma: EAP hyperparameter
%     omega: EAP hyperparameter
%     maxiter: maximum number of iterations
%     conviter: number of iterations without changes in exemplars at time T for convergence
%     burnin: number of iterations without considering consensus node creation. default = 0
%     metric: similarity metric. 'Euclidean', 'Correlation', or 'rbf', default='Euclidean'
%     snMinSize: minimum number of points in cluster to create or continue consensus node. default=2
%     sniter:allow for creation of new supernodes when mod(iter,sniter)==0. default=1
%     sigma2: hyperparameter for rbf metric
%     Outputs: 
%     idx: NxT array. Entry i,t is the exemplar for the cluster to which data point i is assigned at time t.
%           Data points with the same exemplar belong to the same cluster, and exemplars found across time steps
%           allow for tracking the cluster.
%     
    
    if nargin<8
        burnin=0;
    end
    if nargin<9
        metric='Euclidean';
    end
    if nargin<10
        snMinSize=2;
    elseif isempty(snMinSize)
        snMinSize=2;
    end
    if nargin<12 
        sigma2=5;
    end
    if nargin<11
        sniter = 1; 
    end
    S0=S;
    N0=size(S{1},1); %number of total datapoints (don't need to always be alive)
    T=length(S);
    A=reshape(mat2cell(zeros(N0,N0,T),N0,N0,ones(T,1)),T,1,1);
    R=reshape(mat2cell(zeros(N0,N0,T),N0,N0,ones(T,1)),T,1,1); 
    Q=reshape(mat2cell(zeros(N0,N0,T),N0,N0,ones(T,1)),T,1,1);
    D=reshape(mat2cell(zeros(N0,N0,T),N0,N0,ones(T,1)),T,1,1);
    e=zeros(N0,conviter);
    dataNew = data; %will be updating this with supernodes
    dead=cell(T,1); %dead supernodes
    unborn = cell(T,1);  %unborn supernodes
    cont_burnin=1;
    alive = initialize_alive(S,T);
    progress_bar = waitbar(0, 'Passing messages for clustering...');
    for iter=1:maxiter
        waitbar(iter/maxiter, progress_bar,'Passing messages for clustering...');
        for t=1:T
            [A,R,D,Q] = initialize_messages_new_points(alive, t, N0, A, R, D, Q, S);
            alivet = alive{t};
            D = update_message_D(alive, t, N0, omega, gamma, A, R, D, Q, lam);
            R = update_message_R(alive, t, A, R, D, Q, S, lam);
            Q = update_message_Q(alive, t, N0, omega, gamma, A, R, D, Q, lam);
            A = update_message_A(alive, t, A, R, lam);
            if t>1 && ~isempty(dead{t-1})
                Q{t-1}(dead{t-1},:)=0; Q{t-1}(:,dead{t-1})=0;
            end
            if mod(iter,sniter)==0
                [A,R,D,Q,S] = reset_message_dead_unborn(dead, unborn, t, A, R, D, Q, S); %replace messages of dead and unborn nodes with 0
                [E,I,K] = get_exemplars(alive, t, A, R, D, Q);%check if have exemplars
                indNewNodes=[]; snNew =[];
                if K>0 && iter>burnin
                    idt = assign_to_supernode_exemplar(E,I,N0, alivet);
                    [idt, A, R] = clean_up_supernode_assignments(idt, N0, alivet, A, R);
                    [exemp, exempdp, exempsn] = get_final_exemplars(idt, N0, alivet);
                    [indNewNodes, datat, alive, cont_burnin, alivet_prevIter, snNew, snNewInd, dptosn, dptosn_members, numNodesOld, nc] = create_new_supernodes(exempdp, exemp, idt, N0, t, alive, dataNew, A, R, D, Q, cont_burnin, T, snMinSize);
                    [dead, alive, snAlive, A, R, D, Q, S] = update_dead_supernodes(exempsn, alivet_prevIter, alive, dead, datat, idt, t, N0, A, R, D, Q, S, snMinSize);
                    [dataNew, datat] = update_supernode_data(datat, dataNew, alive, t, idt, nc, N0, exemp, exempsn);
                    S = update_similarity(S, t, metric, datat, N0, snAlive, sigma2);
                    [A, R, D, Q]= update_messages_new_supernodes(datat, t, alivet, numNodesOld, indNewNodes, dptosn, N0, A, R, D, Q, S);   
                end
                deadt = dead{t};
                for t1=t+1:T
                    %update supernodes that died
                    %if supernode is dead at time t, it can't be revived.
                    dead{t1} = unique([dead{t1} deadt]); 
                    alive{t1} = [setdiff(alive{t1}, deadt),indNewNodes];
                end
                %create new non-born nodes for previous timesteps to retain order:
                [dataNew,unborn, A,R,D,Q,S] = create_unborn_supernodes_for_order(t, snNew, indNewNodes,dataNew,data,unborn,A,R,D,Q,S);
                if t+1<=T
                    t1=t+1;
                    %modify list of alive nodes for time t1 to include supernodes
                    %alive at t and give the ones that previously died at t1
                    %another chance:
                    numNodesOld = size(dataNew{t+1},1);
                    [alive, dead, alive1, snNotDef] = update_alive_dead_next_t(t, alive, dead, N0, numNodesOld);
                    %update similarity matrix for next time point to include new
                    %supernodes. Do this for all supernodes that haven't died
                    snDead = dead{t1};
                    [S, Stmpa] = update_S_next_t(S,t,dead);
                    datat = dataNew{t}; datat1 = dataNew{t+1};
                    if ~isempty(snNotDef) %if have new supernodes
                        [Atmp,Rtmp,Dtmp,Qtmp]=new_sn_initialize_temp_messages(datat,t,numNodesOld,A,R,D,Q);
                        %update data and messages for new supernodes at time step t
                        datat1a = zeros(size(datat,1),size(datat1,2));
                        datat1a(1:numNodesOld,:) = datat1;
                        [Rtmp, Atmp, Dtmp, Qtmp, alive1, datat1a, singleclust]=update_next_t_new_supernodes(t, snNewInd, dptosn_members, numNodesOld, alive1, datat1, datat1a, R, A, D, Q, N0,Rtmp, Atmp, Dtmp, Qtmp, E);
                        %update similarities to include snNewInd: 
                        S = update_S_next_t_new_supernodes(t, snNewInd, datat1a, alive1, metric, sigma2, Stmpa, numNodesOld, S);
                        %update data for new supernodes
                        if ~isempty(setdiff(snNotDef,snNewInd))
                            new2 = setdiff(snNotDef,snNewInd);
                            for i=1:length(new2)
                                members = find(idt==new2(i));
                                members = intersect(alive1,members); %only consider nodes alive at t+1
                                %find member datapoints in supernode's cluster
                                datat1a(new2(i),:)=mean(datat1(members(members<=N0),:)); %don't consider supernodes
                            end
                            datat1a(snDead,:)=-realmax;
                            dataNew{t+1}=datat1a;
                            %update similarities:
                            S = update_S_undefined_supernodes(t, snNotDef, datat1a, alive1, metric, sigma2, Stmpa, numNodesOld, S);
                            %update message values for new supernodes at time steps
                            %before t using maximum message value from cluster
                            %members
                            [Rtmp, Atmp, Dtmp, Qtmp, singleclust] = initialize_messages_new_supernodes_next_t(t, new2, idt, alivet, numNodesOld, N0, alive1, R, A, D, Q, E, Rtmp, Atmp, Dtmp, Qtmp, singleclust);
                        end
                        if singleclust==1
                            Rtmp = (1-lam)*Rtmp+lam*R{t};
                            Atmp = (1-lam)*Atmp+lam*A{t};
                            Dtmp = (1-lam)*Dtmp+lam*D{t};
                            Qtmp = (1-lam)*Qtmp+lam*Q{t};
                        end
                        %make messages from dead nodes = 0
                        Rtmp(snDead,:)=0; Rtmp(:,snDead)=0;
                        Atmp(snDead,:)=0; Atmp(:,snDead)=0;
                        Dtmp(snDead,:)=0; Dtmp(:,snDead)=0;
                        Qtmp(snDead,:)=0; Qtmp(:,snDead)=0;
                        R{t1}=Rtmp; A{t1}=Atmp;Q{t1}=Qtmp; D{t1}=Dtmp;
                        dataNew{t+1}=datat1a;
                    else %ensure matrix at t+1 is same size as matrix at t, with message values of 0 or realmax
                        [R,A,D,Q,dataNew]=ensure_same_size_t_tplus1(t, dataNew, datat, datat1, numNodesOld,dead, R,A,D,Q);
                    end
                    R{t1}(dead{t1},:)=0; R{t1}(:,dead{t1})=0;
                    A{t1}(dead{t1},:)=0; A{t1}(:,dead{t1})=0;
                    D{t1}(dead{t1},:)=0; D{t1}(:,dead{t1})=0;
                    Q{t1}(dead{t1},:)=0; Q{t1}(:,dead{t1})=0;
                end
            end
        end
        %discard supernodes that are in the union of dead and unborn across all time points
        [R, A, D, Q, S, e, alive, dead, unborn, dataNew] = clean_up_dead_unborn_all_T(T, alive, dead, unborn, R, A, D, Q, S, e,dataNew);
        
        %%% Backwards pass:
        for t=T:-1:1
            [iter t];
            D = update_message_D(alive, t, N0, omega, gamma, A, R, D, Q, lam);
            R = update_message_R(alive, t, A, R, D, Q, S, lam);
            Q = update_message_Q(alive, t, N0, omega, gamma, A, R, D, Q, lam);
            A = update_message_A(alive, t, A, R, lam);
        end
        % Check for convergence based on exemplar assignment at last time step
        ne = length(e);
        e0=e;
        e = zeros(length(A{T}),conviter); e(1:ne,:)=e0;
        E=((diag(A{T}+R{T}+Q{T}+D{T}))>0); e(:,mod(iter-1,conviter)+1)=E; K=sum(E);
        if iter>=burnin+conviter || iter>=maxiter
            se=sum(e,2);
            unconverged=(sum((se==conviter)+(se==0))~=length(e));
            if (~unconverged&&(K>0))||(iter==maxiter)
                waitbar(1, progress_bar, 'Clustering complete.')
                break
            end;
        end;
        nnodes(iter)=length(A{1});
        nnodes(iter);

    end
    idx = assign_exemplars(S0,T,R,A,D,Q,N0,alive);
    fprintf('EAP completed in %d iterations', iter)
    close(progress_bar)
end


function out = onesSNcol(N0,N)
    %output matrix with zeros and ones in supernode columns
    out = zeros(N);
    for k=N0+1:N
        out(:,k)=1;
    end
end


function alive = initialize_alive(S, T)    
    % initialize cell array with indices of alive data points
    alive = cell(T,1);
    for t=1:T
        %consider alive cases those where don't have all 0 or all NaN in row
        %(excluding diagonal entry)
        tmpS = S{t};
        alivetmp = [];
        np=length(tmpS);
        for i=1:np
            if sum(tmpS(i,:)==0)<np-1 && sum(isnan(tmpS(i,:)))<np-1
                alivetmp = [alivetmp,i];
            end
        end
        alive{t}=alivetmp;
    end
end


function [A,R,D,Q] = initialize_messages_new_points(alive, t, N0, A, R, D, Q,S)
    %initialize messages for points inserted at time t based on most similar datapoint
    if t>1
        newPts = setdiff(alive{t},alive{t-1});
        newPts = newPts(newPts<=N0);
        alive1 = alive{t};alive1 = setdiff(alive1,newPts);
        alive1 = alive1(alive1<=N0);
        for j=1:length(newPts)
            new = newPts(j);
            [~,nn1] = max(S{t}(new,alive1));
            nn1=alive1(nn1);
            R{t}(new,:) = R{t}(nn1,:);
            A{t}(new,:) = A{t}(nn1,:);
            D{t}(new,:) = D{t}(nn1,:);
            Q{t}(new,:) = Q{t}(nn1,:);
            R{t}(:,new) = R{t}(:,nn1);
            A{t}(:,new) = A{t}(:,nn1);
            Q{t}(:,new) = Q{t}(:,nn1);
            D{t}(:,new) = D{t}(:,nn1);
            alive2 = setdiff(alive1,new);
            [~,nn2] = max(S{t}(new,alive2));
            nn2=alive2(nn2);
            nn1=alive1(nn1);
            A{t}(new,nn1) = A{t}(nn2,nn1);
            A{t}(nn1,new) = A{t}(nn1,nn2);
            %need to also update diagonal entry of Rtmp and Atmp:
            R{t}(new,new)=R{t}(nn1,nn1);
            A{t}(new,new)=A{t}(nn1,nn1);
            D{t}(new,new)=D{t}(nn1,nn1);
            Q{t}(new,new)=Q{t}(nn1,nn1);
        end
    end
end

function D = update_message_D(alive, t, N0, omega, gamma, A, R, D, Q, lam)
    if t>1 %at first time point, D is zero
        alivet = alive{t};
        alivet1 = alive{t-1};
        common = intersect(alivet, alivet1); %nodes alive at both t and t-1
        indt1 = 1:length(R{t-1});
        keept1 = intersect(indt1,common);
        indt1 = keept1; %since indt1 is 1:N_{t-1}
        Dold = D{t};
        Dnew = Dold;
        R1 = R{t-1}(indt1, indt1);
        A1 = A{t-1}(indt1, indt1);
        Q1 = Q{t-1}(indt1, indt1);
        n0 = find(alivet1<=N0, 1, 'last');
        indsn = onesSNcol(n0,length(R1));
        omegadp=omega*(1-indsn);
        omegasn = omega*indsn;
        c1 = ((gamma-omega)>=R1+A1-Q1);
        c3 = ((-gamma+omegadp)>=+R1+A1-Q1);
        Dtmp = (-gamma+omega).*c1.*c3+(R1+A1-Q1+omegasn).*c1.*(1-c3)+...
            (-R1-A1+Q1).*(1-c1).*c3+(gamma-omegadp).*(1-c1).*(1-c3);
        Dnew(indt1,indt1) = (1-lam)*Dtmp+lam*Dold(indt1,indt1);
        D{t}=Dnew; clear Dtmp Dnew Dold R1 A1 Q1 c1 c3
    end
end

function R = update_message_R(alive, t, A, R, D, Q, S, lam)
    alivet=alive{t}; %defined above
    Atmp = A{t}(alivet,alivet);
    Stmp = S{t}(alivet,alivet);
    Dtmp = D{t}(alivet,alivet);
    Qtmp = Q{t}(alivet,alivet);
    Rold = R{t};
    AS=Atmp+Stmp+Dtmp+Qtmp; [Y,I]=max(AS,[],2);
    for i=1:length(Stmp), AS(i,I(i))=-realmax; end;
    [Y2,~]=max(AS,[],2);
    Rtmp=Stmp+Dtmp+Qtmp-repmat(Y,[1,length(Stmp)]);
    for i=1:length(Rtmp), Rtmp(i,I(i))=Stmp(i,I(i))-Y2(i); end;
    Rtmp=(1-lam)*Rtmp+lam*Rold(alivet,alivet); % Dampen responsibilities
    Rnew = Rold;
    Rnew(alivet,alivet)=Rtmp;
    R{t} = Rnew; 
    clear Rtmp Atmp Dtmp Qtmp Rold Rnew
end

function Q = update_message_Q(alive, t, N0, omega, gamma, A, R, D, Q, lam)
    %update Q(:,t-1). NOTE: Q(:,T)=0
    if t>1
        Qold = Q{t-1};
        alivet = alive{t}; alivet1 = alive{t-1};
        common = intersect(alivet, alivet1);
        indt1 = 1:length(R{t-1});
        [~,keept1,~]=intersect(indt1, common);
        indt1 = indt1(keept1);
        R1 = R{t}(indt1,indt1);
        A1 = A{t}(indt1,indt1);
        D1 = D{t}(indt1,indt1);
        n0 = find(alivet1<=N0, 1, 'last');
        indsn = onesSNcol(n0,length(R1));
        omegadp=omega*(1-indsn);
        omegasn = omega*indsn;
        c1 = ((gamma-omega)>=R1+A1-D1);
        c3 = ((-gamma+omegadp)>=+R1+A1-D1);
        Qtmp = (-gamma+omega).*c1.*c3+(R1+A1-D1+omegasn).*c1.*(1-c3)+...
            (-R1-A1+D1).*(1-c1).*c3+(gamma-omegadp).*(1-c1).*(1-c3);
        Qnew=Qold;
        Qnew(indt1,indt1) = (1-lam)*Qtmp+lam*Qold(indt1,indt1);
        Q{t-1}=Qnew; 
        clear Qtmp Qnew Qold R1 A1 D1 c1 c3
    end
end

function A = update_message_A(alive, t, A, R, lam)
    alivet = alive{t}; %defined above
    Aold=A{t};
    Rtmp = R{t}(alivet,alivet);
    Rp=max(Rtmp,0); for k=1:length(Rp), Rp(k,k)=Rtmp(k,k); end;
    Atmp=repmat(sum(Rp,1),[length(Rp),1])-Rp;
    dA=diag(Atmp);
    Atmp=min(Atmp,0);
    for k=1:length(Atmp), Atmp(k,k)=dA(k); end;
    Anew = Aold;
    Anew(alivet,alivet)=(1-lam)*Atmp+lam*Aold(alivet,alivet); % Dampen availabilities
    A{t}=Anew;
    clear Atmp Aold Anew
end


function [A,R,D,Q,S] = reset_message_dead_unborn(dead, unborn, t, A, R, D, Q, S)
    if ~isempty([dead{t},unborn{t}])
        deadt=dead{t};
        unbornt = unborn{t};
        A{t}([deadt,unbornt],:)=0; A{t}(:,[deadt,unbornt])=0;
        R{t}([deadt,unbornt],:)=0; R{t}(:,[deadt,unbornt])=0;
        D{t}([deadt,unbornt],:)=0; D{t}(:,[deadt,unbornt])=0;
        Q{t}([deadt,unbornt],:)=0; Q{t}(:,[deadt,unbornt])=0;
        S{t}([deadt,unbornt],:)=-realmax; S{t}(:,[deadt,unbornt])=-realmax;
    end
end


function [E,I,K] = get_exemplars(alive, t, A, R, D, Q)
    E=R{t}+A{t}+D{t}+Q{t}; % Pseudomarginals
    alivet=alive{t};
    I=find(diag(E)>0); 
    I = intersect(I,alivet); 
    K=length(I); % Indices of exemplars
end


function idt = assign_to_supernode_exemplar(E,I,N0, alivet)
    %when choosing exemplar, if have supernode as option with E>0, choose
    %supernode (so don't create other sn), otherwise choose datapoint exemplar
    %check if any E(datapoint,supernode)>0 for any datapoints in I:
    Esmall = E(I,I);
    Idp = find(I<=N0); Isn=find(I>N0);
    discardIdp = sum(Esmall(Idp,Isn)>0,2);
    I(Idp(discardIdp>0))=[];
    K=length(I);
    Eg0 = mat2cell(E(:,I)>0,ones(size(E,1),1),length(I));
    insnclust = cellfun(@(x) sum(x(I>N0)>0)>0, Eg0);
    [~, c1]=max(E(:,I),[],2); c1(I)=1:K;
    k1 = find(I>N0,1);
    I2 = I(I>N0);
    if ~isempty(k1)
        [~, c2]=max(E(:,I2),[],2); c2 = c2+k1-1; c2(I2)=k1:K;
    else
        c2 = zeros(size(E,1),1);
    end
    c = c1.*(1-insnclust)+c2.*insnclust;
    idt=I(c); idt = idt(alivet);% Assignments
end


function [idt, A, R] = clean_up_supernode_assignments(idt, N0, alivet, A, R)
    %"clean up" assignments for supernodes that weren't selected
    %if supernode chooses datapoint as exemplar, replace supernode
    %value and messages with those from datapoint.
    %Adjust clustering result to reflect clustering to supernode
    idt_snind = find(idt>N0);
    for k1=1:length(idt_snind)
        if idt(idt_snind(k1))<=N0 %if supernode assigned to datapoint:
            dp1 = idt(idt_snind(k1));
            idt(idt==dp1) = alivet(idt_snind(k1)); %re-assign cluster ids to be those of supernode. 
            %update responsibility and availability message values
            %to be those of datapoint.
            R{t}(alivet(idt_snind(k1)),:) = R{t}(dp1,:);
            A{t}(alivet(idt_snind(k1)),:) = A{t}(dp1,:);
            [~,nn] = max(E(dp1,setdiff(alive1(alive1<=N0),dp1))); 
            if alivet(nn)>=dp1
                nn=nn+1;
            end
            nn=alivet(nn);
            A{t}(alivet(idt_snind(k1)),dp1) = A{t}(dp1, nn); %assign nearest neighbor availability 
            R{t}(:,alivet(idt_snind(k1))) = R{t}(:,dp1);
            A{t}(:,alivet(idt_snind(k1))) = A{t}(:,dp1);
            A{t}(dp1,alivet(idt_snind(k1))) = 0;
            R{t}(alivet(idt_snind(k1)),alivet(idt_snind(k1)))=R{t}(dp1,dp1);
            A{t}(alivet(idt_snind(k1)),alivet(idt_snind(k1)))=A{t}(dp1,dp1);
        end
    end
end


function [exemp, exempdp, exempsn] = get_final_exemplars(idt, N0, alivet)
    %only care about exemplars that have data points (not just
    %supernodes) in cluster
    exemp = unique(idt(alivet<=N0));
    %find exemplars that are NOT supernodes (need to create new
    %supernodes)
    exempdp = exemp(exemp<=N0); %data point exemplars
    exempsn = exemp(exemp>N0); %supernode exemplars
end


function [indNewNodes, datat, alive, cont_burnin, alivet_prevIter, snNew, snNewInd, dptosn, dptosn_members, numNodesOld, nc] = create_new_supernodes(exempdp, exemp, idt, N0, t, alive, dataNew, A, R, D, Q, cont_burnin, T, snMinSize)
    nc = zeros(length(exempdp),1); %number of points in cluster with data point as exemplar
    alivet = alive{t};
    snNew = [];
    dptosn=[]; %data points that lead to supernodes
    dptosn_members=cell(0,1);
    snNewInd = [];
    datat_alive = dataNew{t}(alivet,:);
    if t>1
        numNodest = length(R{t-1}); %consider supernodes from previous time steps
    else
        numNodest = length(R{t});
    end
    if cont_burnin==1
        ctmp = zeros(T,1);
        for t1=1:T
            E1=R{t1}+A{t1}+D{t1}+Q{t1}; % Pseudomarginals
            I=find(diag(E1)>0); 
            if length(I)<2 %need at least 2 clusters in each time step
                cont_burnin=1;
                break;
            else
                ctmp(t1)=1;
            end
        end
        if sum(ctmp)==T
            cont_burnin=0;
        end
    end
    if cont_burnin==0 && length(exemp)>1 && (t==1 || sum(alive{1}>N0)>0)
        for i=1:length(exempdp)
            nc(i)=sum(idt==exempdp(i));
            if nc(i)>=snMinSize %need at least snMinSize (default=2) points in cluster to create supernode
                snNew = [snNew; mean(datat_alive(idt==exempdp(i),:),1)]; %datat_alive only includes data for alive nodes
                snNewInd = [snNewInd, numNodest+1];
                numNodest=numNodest+1;
                dptosn = [dptosn; exempdp(i)];
                dptosn_members= [dptosn_members; {alivet(find(idt==exempdp(i)))}];
            end
        end
    end
    %update data for current time step including new supernodes:
    alivet_prevIter = alive{t}; %supernodes alive at t in previous iteration
    datat = dataNew{t};
    if ~isempty(snNew)
        indNewNodes = [(size(datat,1)+1):size(datat,1)+size(snNew,1)];
        alive{t}=[alive{t} indNewNodes];
    else
        indNewNodes=[];
    end
    numNodesOld = size(datat,1);
    datat=[datat; snNew]; 
end


function [dead, alive, snAlive, A, R, D, Q, S] = update_dead_supernodes(exempsn, alivet_prevIter, alive, dead, datat, idt, t, N0, A, R, D, Q, S, snMinSize)
    snAlive = alivet_prevIter(alivet_prevIter>N0);
    %only assign sn to dead set if have more than one exemplar.
    %don't "kill" supernodes to end up with single cluster
    %declare supernodes as dead if have fewer than snMinSize members
    deadt = dead{t};
    snDeadNew = setdiff(snAlive, exempsn);
    for i=1:length(exempsn)
        if sum(idt==exempsn(i))<snMinSize
            snDeadNew = [snDeadNew, exempsn(i)];
        end
    end
    deadt = [deadt snDeadNew]; %list of dead supernodes at time t
    dead{t} = deadt; %update cell array with dead nodes
    %remove dead nodes from alive set
    alive{t}=setdiff(alive{t}, snDeadNew);
    snAlive = setdiff((N0+1):size(datat,1), deadt);
    %update messages from dead nodes to be NaN:
    S{t}(snDeadNew,:)=-realmax; S{t}(:,snDeadNew)=-realmax;
    A{t}(snDeadNew,:)=0; A{t}(:,snDeadNew)=0;
    R{t}(snDeadNew,:)=0; R{t}(:,snDeadNew)=0;
    D{t}(snDeadNew,:)=0; D{t}(:,snDeadNew)=0;
    Q{t}(snDeadNew,:)=0; Q{t}(:,snDeadNew)=0;
end


function [dataNew, datat] = update_supernode_data(datat, dataNew, alive, t, idt, nc, N0, exemp, exempsn)
    %update existing supernodes for current timestep as mean of
    %data points:
    %Only do this if have more than 1 exemplar so don't have point
    %with average of all data 
    alivet = alive{t};
    if length(exemp)>1
        for i=1:length(exempsn)
            nc(i)=sum(idt==exempsn(i));
            if nc(i)>1 
                datat(exempsn(i),:) = mean(datat(alivet(idt(alivet<=N0)==exempsn(i)),:),1); 
            end
        end
    end
    dataNew{t}=datat; 
end


function S = update_similarity(S, t, metric, datat, N0, snAlive, sigma2)
    %set -realmax as similarity for dead supernodes in current and
    %all subsequent time steps and calculate similarities for new supernodes;
    %update similarities for alive supernodes at time t
    if strcmp(metric, 'Euclidean')
        d = -pdist2([datat(1:N0,:); datat(snAlive,:)],datat(snAlive,:)).^2;
    elseif strcmp(metric, 'Correlation')
        d=corr([datat(1:N0,:); datat(snAlive,:)]',datat(snAlive,:)');
    elseif strcmp(metric, 'rbf')
        d = -pdist2([datat(1:N0,:); datat(snAlive,:)],datat(snAlive,:)).^2;
        d=exp(d/sigma2);
    end
    Stmp2 = -realmax*ones(size(datat,1),size(datat,1));
    Stmp2(1:N0,1:N0)=S{t}(1:N0,1:N0);
    Stmp2([1:N0, snAlive],snAlive) = d;
    Stmp2(snAlive,[1:N0, snAlive]) = d'; %symmetric distance
    for k1 = 1:size(datat,1), Stmp2(k1,k1) = Stmp2(1,1); end %set preferences
    S{t}=Stmp2; %update similarity matrix cell array
end
    

function [A, R, D, Q]= update_messages_new_supernodes(datat, t, alivet, numNodesOld, indNewNodes, dptosn, N0, A, R, D, Q, S)
    if ~isempty(indNewNodes)
        Rtmp = zeros(size(datat,1),size(datat,1));
        Atmp = zeros(size(datat,1),size(datat,1));
        Qtmp = zeros(size(datat,1),1);
        Dtmp = zeros(size(datat,1),1);
        Rtmp(1:numNodesOld,1:numNodesOld)=R{t};
        Atmp(1:numNodesOld,1:numNodesOld)=A{t};
        Qtmp(1:numNodesOld,1:numNodesOld)= Q{t};
        Dtmp(1:numNodesOld,1:numNodesOld)= D{t};
        for k1=1:length(indNewNodes)
            %update messages with values from datapoint exemplar;
            Rtmp(indNewNodes(k1),:) = Rtmp(dptosn(k1),:);
            Atmp(indNewNodes(k1),:) = Atmp(dptosn(k1),:);
            Dtmp(indNewNodes(k1),:) = Dtmp(dptosn(k1),:);
            Qtmp(indNewNodes(k1),:) = Qtmp(dptosn(k1),:);
            [~,nn] = max(S{t}(dptosn(k1),setdiff(alivet(alivet<=N0),dptosn(k1))));
            if alivet(nn)>=dptosn(k1)
                nn=nn+1;
            end
            nn=alivet(nn);
            Atmp(indNewNodes(k1),dptosn(k1)) = Atmp(dptosn(k1), nn); %assign nearest neighbor availability 
            Rtmp(:,indNewNodes(k1)) = Rtmp(:,dptosn(k1));
            Atmp(:,indNewNodes(k1)) = Atmp(:,dptosn(k1));
            Dtmp(:,indNewNodes(k1)) = Dtmp(:,dptosn(k1));
            Qtmp(:,indNewNodes(k1)) = Qtmp(:,dptosn(k1));

            Atmp(dptosn(k1),indNewNodes(k1)) = 0;
            Rtmp(indNewNodes(k1),indNewNodes(k1))=Rtmp(dptosn(k1),dptosn(k1)); 
            Atmp(indNewNodes(k1),indNewNodes(k1))=Atmp(dptosn(k1),dptosn(k1)); 
            Qtmp(indNewNodes(k1),indNewNodes(k1))=Qtmp(dptosn(k1),dptosn(k1));
            Dtmp(indNewNodes(k1),indNewNodes(k1))=Dtmp(dptosn(k1),dptosn(k1));
        end
        R{t}=Rtmp; A{t}=Atmp;Q{t}=Qtmp; D{t}=Dtmp;
    end
end


function [dataNew,unborn, A,R,D,Q,S] = create_unborn_supernodes_for_order(t, snNew, indNewNodes,dataNew,data,unborn,A,R,D,Q,S)
    %create new non-born nodes for previous timesteps to retain order:
    if t>1 && ~isempty(snNew)
        lnew = length(indNewNodes);
        for t1=1:t-1
            Stmp = S{t1}; S1 = zeros(length(Stmp)+lnew, length(Stmp)+lnew);
            S1(1:length(Stmp), 1:length(Stmp))=Stmp;
            S{t1}=S1;
            Atmp = zeros(length(Stmp)+lnew, length(Stmp)+lnew);
            Atmp(1:length(Stmp), 1:length(Stmp))=A{t1};
            A{t1}=Atmp;
            Rtmp = zeros(length(Stmp)+lnew, length(Stmp)+lnew);
            Rtmp(1:length(Stmp), 1:length(Stmp))=R{t1};
            R{t1}=Rtmp;
            Dtmp = zeros(length(Stmp)+lnew, length(Stmp)+lnew);
            Dtmp(1:length(Stmp), 1:length(Stmp))=D{t1};
            D{t1}=Dtmp;
            Qtmp = zeros(length(Stmp)+lnew, length(Stmp)+lnew);
            Qtmp(1:length(Stmp), 1:length(Stmp))=Q{t1};
            Q{t1}=Qtmp;
            %update dataNew{t} at previous time steps
            dataNew{t1}=[dataNew{t1}; zeros(lnew,size(data{t1},2))];
            %add list of new nodes to unborn array at previous time steps
            unborn{t1} = [unborn{t1}, indNewNodes];
        end
    end
end


function [alive, dead, alive1, snNotDef] = update_alive_dead_next_t(t, alive, dead, N0, numNodesOld)
    %modify list of alive nodes for time t1 to include supernodes
    %alive at t and give the ones that previously died at t1
    %another chance:
    t1=t+1;
    snNotDef = setdiff(alive{t1}(alive{t1}>N0), 1:numNodesOld); %find list of nodes that haven't been added to data
    alive1=alive{t1};
    snNotDef = [snNotDef, setdiff(alive{t}(alive{t}>N0),alive1(alive1>N0))]; %add nodes alive at t but not t1
    alive{t1}=union(alive{t}(alive{t}>N0),alive{t1});
    dead{t1} = setdiff(dead{t1},alive{t1});
end


function [S, Stmpa] = update_S_next_t(S,t,dead)
    t1=t+1;
    Stmp = S{t1};
    Stmpa = zeros(size(S{t}));
    Stmpa(1:length(Stmp), 1:length(Stmp))=Stmp;
    snDead = dead{t1};
    Stmpa(snDead,:)=-realmax;
    Stmpa(:,snDead)=-realmax;
    S{t1}=Stmpa; 
end


function [Atmp,Rtmp,Dtmp,Qtmp]=new_sn_initialize_temp_messages(datat,t,numNodesOld,A,R,D,Q)
    %initialize messages for existing nodes to be previous message values
    t1=t+1;
    Rtmp = zeros(size(datat,1),size(datat,1));
    Atmp = zeros(size(datat,1),size(datat,1));
    Qtmp = zeros(size(datat,1),size(datat,1));
    Dtmp = zeros(size(datat,1),size(datat,1));
    Rtmp(1:numNodesOld,1:numNodesOld)=R{t1};
    Atmp(1:numNodesOld,1:numNodesOld)=A{t1};
    Qtmp(1:numNodesOld,1:numNodesOld)= Q{t1};
    Dtmp(1:numNodesOld,1:numNodesOld)= D{t1};
end


function [Rtmp, Atmp, Dtmp, Qtmp, alive1, datat1a, singleclust]=update_next_t_new_supernodes(t, snNewInd, dptosn_members, numNodesOld, alive1, datat1, datat1a, R, A, D, Q, N0,Rtmp, Atmp, Dtmp, Qtmp, E)
    snNewRmv=[];
    t1=t+1;
    singleclust=0;
    for i=1:length(snNewInd)
        memb = dptosn_members{i}; %only contains datapoints
        memb = memb(memb<=numNodesOld); 
        memb = intersect(memb, alive1);
        if ~isempty(memb)
            %from prev timestep is in new sn cluster
            datat1a(snNewInd(i),:)=mean(datat1(memb,:));
            %for message update, also consider current datapoint exemplars at
            %t1 for memb:
            E1=R{t1}+A{t1}+D{t1}+Q{t1}; % Pseudomarginals
            I1=find(diag(E1)>0); K1=length(I1); % Indices of exemplars
            if K1==1
                singleclust=1;
            end
            if ~isempty(I1)
                [~, c]=max(E1(:,I1),[],2); c(I1)=1:K1;
                idt1 = I1(c);
                indmax = mode(idt1(memb)); %get message values from most common exemplar
            else
                indmax = NaN;
            end
            if indmax>N0 || isnan(indmax) 
                snNewRmv=[snNewRmv,snNewInd(i)];
            else
                Rtmp(snNewInd(i),:) = Rtmp(indmax,:);
                Atmp(snNewInd(i),:) = Atmp(indmax,:); 
                Dtmp(snNewInd(i),:) = Dtmp(indmax,:);
                Qtmp(snNewInd(i),:) = Qtmp(indmax,:);
                [~,nn] = max(E(indmax,setdiff(alive1(alive1<=N0),indmax))); 
                if alive1(nn)>=indmax && ~isempty(intersect(alive1,indmax))
                    nn=nn+1;
                end
                nn=alive1(nn);
                Atmp(snNewInd(i),indmax) = Atmp(indmax, nn); %assign nearest neighbor availability 
                Rtmp(:,snNewInd(i)) = Rtmp(:,indmax);
                Atmp(:,snNewInd(i)) = Atmp(:,indmax); 
                Qtmp(:,snNewInd(i)) = Qtmp(:,indmax);
                Dtmp(:,snNewInd(i)) = Dtmp(:,indmax);
                Atmp(indmax,snNewInd(i)) = 0;
                %update diagonal entry of Rtmp and Atmp:
                Rtmp(snNewInd(i),snNewInd(i))=Rtmp(indmax,indmax);
                Atmp(snNewInd(i),snNewInd(i))=Atmp(indmax,indmax);
                Dtmp(snNewInd(i),snNewInd(i))=Dtmp(indmax,indmax);
                Qtmp(snNewInd(i),snNewInd(i))=Qtmp(indmax,indmax);
            end
        else
            snNewRmv=[snNewRmv,snNewInd(i)];
        end
    end
end


function S = update_S_next_t_new_supernodes(t, snNewInd, datat1a, alive1, metric, sigma2, Stmpa, numNodesOld, S)
    if ~isempty(snNewInd)
        if strcmp(metric, 'Euclidean')
            d = -pdist2([datat1a(alive1(alive1<=numNodesOld),:); datat1a(snNewInd,:)],datat1a(snNewInd,:)).^2; 
        elseif strcmp(metric, 'Correlation')
            d = corr([datat1a(alive1(alive1<=numNodesOld),:); datat1a(snNewInd,:)]',datat1a(snNewInd,:)');
        elseif strcmp(metric, 'rbf')
            d = -pdist2([datat1a(alive1(alive1<=numNodesOld),:); datat1a(snNewInd,:)],datat1a(snNewInd,:)).^2; 
            d = exp(d/sigma2);
        end
        Stmp2 = Stmpa;
        Stmp2([alive1(alive1<=numNodesOld), snNewInd],snNewInd) = d;
        Stmp2(snNewInd,[alive1(alive1<=numNodesOld), snNewInd]) = d'; %symmetric distance
        S{t+1}=Stmp2;
    end
end


function S = update_S_undefined_supernodes(t, snNotDef, datat1a, alive1, metric, sigma2, Stmpa, numNodesOld, S)
    if strcmp(metric, 'Euclidean')
        d = -pdist2([datat1a(alive1(alive1<=numNodesOld),:); datat1a(snNotDef,:)],datat1a(snNotDef,:)).^2; 
    elseif strcmp(metric, 'Correlation')
        d = corr([datat1a(alive1(alive1<=numNodesOld),:); datat1a(snNotDef,:)]',datat1a(snNotDef,:)'); 
    elseif strcmp(metric, 'rbf')
        d = -pdist2([datat1a(alive1(alive1<=numNodesOld),:); datat1a(snNotDef,:)],datat1a(snNotDef,:)).^2;
        d = exp(d/sigma2);
    end

    Stmp2 = Stmpa;
    Stmp2([alive1(alive1<=numNodesOld), snNotDef],snNotDef) = d;
    Stmp2(snNotDef,[alive1(alive1<=numNodesOld), snNotDef]) = d'; %symmetric distance
    S{t+1}=Stmp2;
end

function [Rtmp, Atmp, Dtmp, Qtmp, singleclust] = initialize_messages_new_supernodes_next_t(t, new2, idt, alivet, numNodesOld, N0, alive1, R, A, D, Q, E, Rtmp, Atmp, Dtmp, Qtmp, singleclust)
    t1=t+1;
    for k1=1:length(new2)
        memb = find(idt==new2(k1));
        memb = alivet(memb)'; %idt only includes alive nodes
        memb = memb(memb<=numNodesOld); 
        memb = intersect(memb, alive1);
        E1=R{t1}+A{t1}+D{t1}+Q{t1}; % Pseudomarginals
        I1=find(diag(E1)>0); K1=length(I1); % Indices of exemplars
        if K1==1
            singleclust=1;
        end
        if ~isempty(I1)
            [~, c]=max(E1(:,I1),[],2); c(I1)=1:K1;
            idt1 = I1(c);
            addmemb = unique(idt1(memb));
            if ~isempty(setdiff(addmemb, memb))
                1;
            end
        else
            idt1=idt;
        end
        indmax = mode(idt1(memb)); 
        try Rtmp(new2(k1),:) = Rtmp(indmax,:);
        catch
            pass
        end
        Atmp(new2(k1),:) = Atmp(indmax,:); 
        Dtmp(new2(k1),:) = Dtmp(indmax,:);
        Qtmp(new2(k1),:) = Qtmp(indmax,:);
        [~,nn] = max(E(indmax,setdiff(alive1(alive1<=N0),indmax))); 
        if alive1(nn)>=indmax && ~isempty(intersect(alive1,indmax))
            nn=nn+1;
        end
        nn=alive1(nn);
        Atmp(new2(k1),indmax) = Atmp(indmax, nn); 
        Rtmp(:,new2(k1)) = Rtmp(:,indmax);
        Atmp(:,new2(k1)) = Atmp(:,indmax); 
        Dtmp(:,new2(k1)) = Dtmp(:,indmax);
        Qtmp(:,new2(k1)) = Qtmp(:,indmax);
        Atmp(indmax,new2(k1)) = 0;
        %update diagonal entries
        Rtmp(new2(k1),new2(k1)) = Rtmp(indmax,indmax);
        Atmp(new2(k1),new2(k1)) = Atmp(indmax,indmax);
        Dtmp(new2(k1),new2(k1)) = Dtmp(indmax,indmax);
        Qtmp(new2(k1),new2(k1)) = Qtmp(indmax,indmax);
    end
end


function [R,A,D,Q,dataNew]=ensure_same_size_t_tplus1(t, dataNew, datat, datat1, numNodesOld,dead, R,A,D,Q)
    t1=t+1;
    if length(dataNew{t})>length(dataNew{t+1})
        %create data points with 0 and message matrices with
        %dummy values
        Rtmp = zeros(size(datat,1),size(datat,1));
        Atmp = zeros(size(datat,1),size(datat,1));
        Qtmp = zeros(size(datat,1),size(datat,1));
        Dtmp = zeros(size(datat,1),size(datat,1));
        Rtmp(1:numNodesOld,1:numNodesOld)=R{t1};
        Atmp(1:numNodesOld,1:numNodesOld)=A{t1};
        Dtmp(1:numNodesOld,1:numNodesOld)=D{t1};
        Qtmp(1:numNodesOld,1:numNodesOld)=Q{t1};
        %update data and messages for new supernodes at time step t
        datat1a = zeros(size(datat,1),size(datat1,2));
        datat1a(1:numNodesOld,:) = datat1;
        snDead = dead{t1};
        Rtmp(snDead,:)=0; Rtmp(:,snDead)=0;
        Atmp(snDead,:)=0; Atmp(:,snDead)=0;
        Dtmp(snDead,:)=0; Dtmp(:,snDead)=0;
        Qtmp(snDead,:)=0; Qtmp(:,snDead)=0;
        R{t1}=Rtmp; A{t1}=Atmp; Q{t1}=Qtmp; D{t1}=Dtmp;
        dataNew{t+1}=datat1a;
    end
end


function [R, A, D, Q, S, e, alive, dead, unborn, dataNew] = clean_up_dead_unborn_all_T(T, alive, dead, unborn, R, A, D, Q, S, e,dataNew)
    deadAllT = dead{1};
    unborn1 = unborn{1};
    indrmv = [deadAllT,unborn1];
    if ~isempty(indrmv)
        for t=1:T
            indrmv = intersect(indrmv, [dead{t}, unborn{t}]);
        end
        if ~isempty(indrmv)
            for t=1:T
                deadt = dead{t};
                alivet=alive{t};
                unbornt = unborn{t};
                Atmp = A{t}; Atmp(indrmv,:)=[]; Atmp(:,indrmv)=[]; A{t}=Atmp;
                Rtmp = R{t}; Rtmp(indrmv,:)=[]; Rtmp(:,indrmv)=[]; R{t}=Rtmp;
                Stmp = S{t}; Stmp(indrmv,:)=[]; Stmp(:,indrmv)=[]; S{t}=Stmp;
                Dtmp = D{t}; Dtmp(indrmv,:)=[]; Dtmp(:,indrmv)=[]; D{t}=Dtmp;
                Qtmp = Q{t}; Qtmp(indrmv,:)=[]; Qtmp(:,indrmv)=[]; Q{t}=Qtmp;
                datat = dataNew{t}; datat(indrmv,:)=[]; dataNew{t}=datat;
                if t==T
                    %update convergence matrix
                    e(indrmv,:)=[];
                end
                %update list of dead and unborn nodes
                [~,indd,~]=intersect(deadt,indrmv);
                deadt(indd)=[];
                [~,indu,~]=intersect(unbornt,indrmv);
                unbornt(indu)=[];
                %correct numbering of deadt, alivet, and unbornt
                for k=length(indrmv):-1:1
                    deadt(deadt>indrmv(k))= deadt(deadt>indrmv(k))-1;
                    alivet(alivet>indrmv(k))= alivet(alivet>indrmv(k))-1;
                    unbornt(unbornt>indrmv(k)) = unbornt(unbornt>indrmv(k))-1;
                end
                dead{t}=deadt; %update set of dead supernodes
                alive{t}=alivet;
                unborn{t}=unbornt;
            end
        end
    end
end


function idx = assign_exemplars(S0,T,R,A,D,Q,N0,alive)
    idx=zeros(length(S0{1}),T);
    for t1a=1:T
        E=R{t1a}+A{t1a}+D{t1a}+Q{t1a}; % Pseudomarginals
        I=find(diag(E)>0); 
        K=length(I); % Indices of exemplars
        %when choosing exemplar, if have supernode as option with E>0, choose
        %supernode (don't create other sn), otherwise choose datapoint exemplar
        %check if any E(datapoint,supernode)>0 for any datapoints in I:
        if ~isempty(I)
            Esmall = E(I,I);
            Idp = find(I<=N0); Isn=find(I>N0);
            discardIdp = sum(Esmall(Idp,Isn)>0,2);
            I(Idp(discardIdp>0))=[];
            K=length(I);
            Eg0 = mat2cell(E(:,I)>0,ones(size(E,1),1),length(I));%logical indicating which E values > 0
            insnclust = cellfun(@(x) sum(x(I>N0)>0)>0, Eg0);
            [~, c1]=max(E(:,I),[],2); c1(I)=1:K;
            k1 = find(I>N0,1);
            I2 = I(I>N0);
            if ~isempty(k1)
                [~, c2]=max(E(:,I2),[],2); c2 = c2+k1-1; c2(I2)=k1:K;
            else
                c2 = zeros(size(E,1),1);
            end
            c = c1.*(1-insnclust)+c2.*insnclust;
            idt1=I(c);
            idt0 = nan(N0,1); idt0(alive{t1a}(alive{t1a}<=N0))=idt1(alive{t1a}(alive{t1a}<=N0));
            idx(:,t1a)=idt0;
        end
    end
end