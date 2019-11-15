input_data = load('sounds.mat');
%input_data = load('icaTest.mat');
U = input_data.sounds(1:5,:);
% 1- male voice 2- grinder 3- clap 4- laugh 5- paper crushing
m = size(U,1);
n = size(U,1);
%A = rand(n); % m x n matrix
a = -30;% figure
% for i = 1:n
%     subplot(n,1,i);
%     plot(U(i,1:1000));    
% end
% sgtitle('Original signals')

b = 30;
A = a + (b-a).*rand(n,n);
%A = input_data.A;
X = A * U;
W = rand(n);
learning_rate = 0.05;
iterations = 100000;
I = eye(n,n);
for i = 1:iterations
    Y = W * X;
    Z = 1.0 ./ (1.0 + exp(-Y));
    delta_W = learning_rate * (I + (1-2*Z) * Y'/44000)* W;
    W = W + delta_W;
    error = delta_W * X;    
    if(abs(error) < 10^-6)
        break
    end
end
% figure
% for i = 1:n
%     subplot(n,1,i);
%     plot(U(i,1:1000));    
% end
% sgtitle('Original signals')

% figure
% for i = 1:n
%      subplot(n,1,i);
%      plot(X(i,1:1000));    
% end
% sgtitle('Mixed signals')

% figure
% for i = 1:n
%     subplot(n,1,i)
%     plot(Y(i,1:1000))
% end
% sgtitle('Recovered signals')
x_axis = linspace(0,200,200);
for i = 1:n
    accuracy = 0;
    for j = 1:n
        corr_coef = abs(corrcoef(U(i,:),Y(j,:)));
        max_acc = min(corr_coef(1,:));
        if max_acc > accuracy
            input_signal = i;
            output_signal = j;
            accuracy = max_acc;
        end
    end
    fprintf('I/p index - %d, O/p index - %d, accuracy - %d\n', input_signal, output_signal,accuracy);    
    ip = rescale(U(input_signal,1:200),-1,1);
    mix = rescale(X(i,1:100),-1,1);
    op = rescale(Y(output_signal,1:200),-1,1);    
%     if input_signal==2 || input_signal==4 || input_signal==5
        figure
        subplot(2,1,1)
        plot(x_axis,ip,'r')
        subplot(2,1,2)
        plot(x_axis,op,'b')
%     end
end
