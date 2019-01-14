% demo_sparseSVM.m
%
% Sparse Support Vector Machine (sparseSVM) for imbalanced class sizes  
%
% J. Frecon. Version: 03-March-2017.

clc;
clear all;

mydir  = which('demo_sparseSVM.m');
idcs   = strfind(mydir,'/');
newdir = mydir(1:idcs(end));       
addpath(genpath(newdir));


sparseSVM_perf  = @(w,b,y) sign(w'*y - b);


%% - Training set
K           = 15;                           % #features
N           = 500;                          % #subjects
cst         = 10;
sigma       = .5;
x           = sigma*randn(K,N)+cst;
shift       = randn(K,1);                   % shift between classes
example     = 2;


% Assignement of labels
switch example                              
    case 1  % Balanced case
        z   = sign(randn(1,N));
    case 2  % Imbalanced case
        frac = 0.05;
        N1 = fix(frac*N);
        N2 = N-N1;
        z   = [ones(1,N1), -ones(1,N2)];
end
x(:,z==1)  = x(:,z==1) + shift*ones(1,length(x(:,z==1)));



%% - Sparse SVM

% Preprocessing
mu0 = mean(x');
s0  = std(x');
x   = (x-mu0'*ones(1,N))./(s0'*ones(1,N));

% Computation
tic
C      = .1;                           % Trade-off between data fidelity & sparsity
[w, b] = sparseSVM(x,z,C);
toc

% Results
disp('Normal vector: w')
disp(w)
disp('Offset: b');
disp(b);


%% - Performance

% Test set
Nperf           = 100;
y               = sigma*randn(K,Nperf)+cst;
zbis            = sign(randn(1,Nperf));
y(:,zbis==1)    = y(:,zbis==1) + shift*ones(1,length(y(:,zbis==1)));

% Preprocessing
y = (y-mu0'*ones(1,Nperf))./(s0'*ones(1,Nperf));

% Computation & result
[zest]          = sparseSVM_perf(w,b,y);
disp(strcat('Number of correct classifications: ',num2str(sum(zest==zbis)),'/',num2str(length(zbis))));



%% - Display

% Display feature 'dispFeature1' against 'dispFeature2'
dispFeature1 = 3;                   
dispFeature2 = 5;

figure(1); clf;
set(gca,'fontsize',15);
bar(w,'b','linewidth',2); hold on;
xlabel('Features','Interpreter','latex');
ylabel('Normal vector $w$','Interpreter','latex');
grid on;


test    = [min(min(x(dispFeature1,:),x(dispFeature2,:))):.1:max(max(x(dispFeature2,:),x(dispFeature1,:)))];
alpha   = -w(dispFeature1)/w(dispFeature2);
beta    = sqrt(alpha^2+1)*b;
sgn     = sign(mean(x(dispFeature2,z==1))-mean(x(dispFeature2,z==-1)));
y_HP    = alpha*test + sgn*beta;    %Hyperplane equation


figure(2); clf;
set(gca,'fontsize',15);
gscatter(x(dispFeature1,:),x(dispFeature2,:),z); hold on;
xlabel(strcat('$x_',num2str(dispFeature1),'$'),'Interpreter','latex');
ylabel(strcat('$x_',num2str(dispFeature2),'$'),'Interpreter','latex');
axis([min(x(dispFeature1,:)) max(x(dispFeature1,:)) min(x(dispFeature2,:)) max(x(dispFeature2,:))]);
plot(test,y_HP,'-k');
grid on;



%%


figure(1); clf;
bar(w,'b','linewidth',2); hold on;
xlabel('Features','Interpreter','latex','fontsize',20);
ylabel('Normal vector $w$','Interpreter','latex','fontsize',20);
grid on;
set(gca,'fontsize',20);

test    = [min(min(x(dispFeature1,:),x(dispFeature2,:))):.1:max(max(x(dispFeature2,:),x(dispFeature1,:)))];
alpha   = -w(dispFeature1)/w(dispFeature2);
beta    = sqrt(alpha^2+1)*b;
sgn     = sign(mean(x(dispFeature2,z==1))-mean(x(dispFeature2,z==-1)));
y_HP    = alpha*test + sgn*beta;    %Hyperplane equation


figure(2); clf;
gscatter(x(dispFeature1,:),x(dispFeature2,:),z); hold on;
xlabel('Feature 1','Interpreter','latex','fontsize',20);
ylabel('Feature 2','Interpreter','latex','fontsize',20);
%xlabel(strcat('$x_',num2str(dispFeature1),'$'),'Interpreter','latex');
%ylabel(strcat('$x_',num2str(dispFeature2),'$'),'Interpreter','latex');
axis([min(x(dispFeature1,:)) max(x(dispFeature1,:)) min(x(dispFeature2,:)) max(x(dispFeature2,:))]);
plot(test,y_HP,'-k');
grid on;
set(gca,'fontsize',20);