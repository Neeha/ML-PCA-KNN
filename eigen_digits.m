input_data = load('digits (3).mat');
train_images = input_data.trainImages;
train_labels = input_data.trainLabels;
test_images = input_data.testImages;
test_labels = input_data.testLabels;
image_dims = [28, 28];
r=28;
c=28;
image_size = 784; %28 * 28
train_size = 10000;
test_size = 1000;
[A, X] = imagesToVectors(train_images, test_images,image_size, train_size,test_size);
[m, V] = hw1FindEigenDigits(A);
%displayEigenVectors(V);
top_n = 100;
[A_proj, X_proj] = imageProjection(A,X,m,V,train_size, test_size,top_n);

%RECONSTRUCTION OF A SINGLE IMAGE
num = randi([1 10000]);
im = im2double(train_images(:,:,1,num));
Vtemp = (V(:,1:top_n));
I_recon = (A_proj(num,:)*Vtemp') + m';
I_recon = reshape(I_recon, r,c);

%display original and reconstructed test image
figure
subplot(1,2,1);
imshow(im);
title('Original image');
subplot(1,2,2)
imshow(I_recon);
title('Reconstructed image');
  
computeAccuracy_Ksize(A_proj,X_proj, train_labels,test_labels,train_size,test_size);
computeAccuracy_eigenVectorsSize(A,X,m,V,train_labels,test_labels,train_size,test_size);

%COMPUTING ACCURACY FOR TOP N EIGEN VECTORS BY VARYING K
function computeAccuracy_Ksize(A_proj,X_proj, train_labels, test_labels,train_size, test_size)    
    predicted_label = zeros(test_size,1);
    %the K values for the KNN model
    k_values=[3,5];
    ks=zeros(length(k_values),1);
    acc=zeros(length(k_values),1);
    for j = 1:length(k_values)
        k = k_values(j);
        KNNmdl = fitcknn(A_proj,train_labels(:,1:train_size),'NumNeighbors',k,'Standardize',1)
        for i = 1:test_size
            Xnew = X_proj(i,:);
            label = predict(KNNmdl,Xnew);
            predicted_label(i) = label;
        end
        %calculating the number of correct predictions and accuracy
        num_correct = sum(test_labels(:,1:test_size)==predicted_label');
        accuracy = num_correct / test_size(1);
        sprintf("k: %d, Acc:%f",k, accuracy)
        ks(j) = k;
        acc(j) = accuracy;
    end
    figure, plot(ks,acc);
    xlabel('K'), ylabel('Accuracy');
end

%COMPUTING ACCURACY BY VARYING NO. OF EIGEN VECTORS FOR K=3
function computeAccuracy_eigenVectorsSize(A,X,m,V, train_labels,test_labels, train_size,test_size)
    predicted_label = zeros(test_size,1);
    %ev=[10 50 60 75 90 100 200 300 500 600 784];
    ev=[10 50 100 500 784];
    no_of_ev=zeros(length(ev),1);
    acc=zeros(length(ev),1);
    for j = 1:length(ev)
        [A_proj, X_proj] = imageProjection(A,X,m,V,train_size, test_size,ev(j));
        KNNmdl = fitcknn(A_proj,train_labels(:,1:train_size),'NumNeighbors',3,'Standardize',1);
        for i = 1:test_size
            Xnew = X_proj(i,:);
            label = predict(KNNmdl,Xnew);
            predicted_label(i) = label;
        end
        num_correct = sum(test_labels(:,1:test_size)==predicted_label');
        accuracy = num_correct / test_size(1);
        sprintf("N: %d, Acc:%f",ev(j), accuracy)
        no_of_ev(j) = ev(j);
        acc(j) = accuracy;
    end
    figure, plot(no_of_ev,acc);
    xlabel('No. of eigenvectors'), ylabel('Accuracy');    
end

function [A,X] = imagesToVectors(train_images, test_images,image_size,train_size,test_size)
    %converting train images to a vector
    A = zeros(image_size, train_size);    
    for i = 1:train_size
        img = train_images(:,:,1,i);
        imgvector = reshape(img, [], 1);
        A(:,i) = imgvector;        
    end
    
    %converting test images to a vector
    X = zeros(image_size, test_size);
    for i = 1:test_size
        img = test_images(:,:,1,i);
        imgvector = reshape(img, [], 1);
        X(:,i) = imgvector;        
    end
end
function [m,V] = hw1FindEigenDigits(A)
    m = mean(A,2);    
    %mean subtraction
    A = A-m;
    A = A';
    %since cov(A) considers columns represent random variables and rows 
    %represent observations, finding the transpose of A    
    coVariance = cov(A);
    [eigen_vectors,eigen_values] = eig(coVariance);
    
    %sorting eigen vectors in descending order
    [eigen_values_sorted,sorted_order] = sort(diag(eigen_values), 'descend');
    V = eigen_vectors(:, sorted_order);
    
    %normalizing the eigen vectors
    for i = 1:size(V,2)
        V(:,i) = V(:,i)/norm(V(:,i));
    end
    displayEigenValues(eigen_values_sorted);
end
function [A_proj, X_proj] = imageProjection(A,X,m,V,train_size, test_size,top_n)
    %Choosing only the top n eigen vectors
    V = (V(:,1:top_n));
    
    %projecting the training images on the reduced(n) eigen space
    A_shifted = A-repmat(m,1,train_size);        
    A_proj = A_shifted' * V;
    
    %projecting the test images on the reduced(n) eigen space
    X_shifted = X -repmat(m,1,test_size);
    X_proj = X_shifted' * V;    
end
function displayEigenValues(evalues)
    normalised_evalues = evalues / sum(evalues);
    figure, plot(cumsum(normalised_evalues));
    xlabel('Eigen Values'), ylabel('Variance accounted for');
    %xlim([1 784]), ylim([0 1]), grid on;
end
function displayEigenVectors(evectors)
    figure;
    sample_eigen_numbers = 8;
    image_dims = [28, 28];
    for n = 1:sample_eigen_numbers
        subplot(2, ceil(sample_eigen_numbers/2), n);
        evector = reshape(evectors(:,n), image_dims);
        imshow(evector);
    end
end
