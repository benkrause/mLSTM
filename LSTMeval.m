function LSTMeval

gpuDevice(1)

sequence = processtextfile('enwik8');

weightsfname = 'LSTMhutter.mat';

nunits = 205;


network = initnetwork(nunits,[1250,1250],nunits);

    network.last = 0;




    network=updateV(network,-1*weights2vect(getW(network)));

  wfile = load(weightsfname,'W');
    W = wfile.W;

 
    network = updateV(network,W);

disp(network.nparam)






 seqlen = 50;





megabatch = 1000000;
minibatch = 10000;
jit = megabatch/minibatch;

;
tic

   start = 95000000;;

    fin = start+1*megabatch-1;
  
    
  
    
    serr = 0; 
  
    
    network.last = 0;
    nbatch = minibatch/seqlen;
  
 
    for k=1:5
        
        
          [in0,ta0] = byte2input(sequence(start:fin),sequence(start+1:fin+1),nunits,seqlen);
          
    start = start+megabatch;
    fin = fin+megabatch;
          
          
    
    for j=1:jit

    
  
    in1= in0(:,j:jit:size(in0,2),:);
    ta1= ta0(:,j:jit:size(ta0,2),:);
   
    
    
  
  
 
 
  
    
    
    
      network.zero = 1;

    network = initpass(network,size(in1,2),size(in1,3));
  

    network = ForwardPass(network,in1);
    network.output.outs.v = .999999*network.output.outs.v + .000001*(ones(size(network.output.outs.v)))/size(network.output.outs.v,1);
    
  
    
    terr = evalCrossEntropy(network.output.outs.v,ta1,ones(size(ta1),'single'));
   
    serr = serr+ (terr/(nbatch*seqlen));
    
    err = serr/(jit*(k-1)+j);

  
          network.last = 1;
  
     if mod(j,100)==0
      
         disp(err)
     end

    end

    end
    disp('final error')
    disp(err)
end
function [inputs,targets] =byte2input(inputs,targets,nunits,seqlen)


    inputs= single(inputs);
    targets = single(targets);

    in = zeros(nunits,1,length(inputs),'single');
    targ = zeros(nunits,1,length(targets),'single');
   ind = sub2ind([nunits,1,length(inputs)],inputs,ones(length(inputs),1),(1:length(inputs))');
   tind = sub2ind([nunits,1,length(inputs)],targets,ones(length(inputs),1),(1:length(inputs))');
    in(ind)=1;
   targ(tind) = 1;
   
  inputs=permute(reshape(in,[size(in,1),seqlen,size(in,3)/seqlen]),[1,3,2]);
   targets=permute(reshape(targ,[size(targ,1),seqlen,size(targ,3)/seqlen]),[1,3,2]);
  
   
  

end



function gradient = getGradient(network,inputs,targets,npar)

   
           pbatch = size(inputs,2)/npar;
            citer = 1:pbatch:size(inputs,2);
            oind = ones(size(targets));
             in = cell(npar,1);ta = cell(npar,1);oi = cell(npar,1);
             for ci=1:length(citer)
            c = citer(ci);
            in{ci} = inputs(:,c:c+pbatch-1,:);
            ta{ci} = targets(:,c:c+pbatch-1,:);
            oi{ci} = oind(:,c:c+pbatch-1,:);
            end
    
              gradient = zeros(network.nparam,1);
                 for z=1:npar
                
                net = initpass(network,size(in{z},2),size(in{z},3));
                net = ForwardPass(net,in{z});
                net = computegradient(net,ta{z},ones(size(ta{z})));
      
                
                   gradient = gradient + weights2vect(getJ(net));
                 end
           
           
        

end

function [err] = test(network,inputs,targets,oind)
        errsum = 0;
        errcount = 0;

        nbatch = size(inputs,2);
        
            
            input = inputs;
            network = initpasstest(network,nbatch,size(input,3));
            network = ForwardPasstest(network,input);
           [terr]=network.errorFunc(network.output.outs.v,targets,oind);
            errsum = errsum + terr;
          
            errcount = errcount+1;
            
            
        
        

      
        
        
       err = errsum/errcount;
 
       
            
    


end




function network = ForwardPass(network, inputs)

inputs = gpuArray(inputs);
network.input.outs.v=inputs;



    s=1;
for t=1:size(inputs,3);
    for l=1:length(network.hidden);
    

    
    
    network.hidden(l).ins.v(:,:,t)= network.hidden(l).iweights.matrix*network.input.outs.v(:,:,t);
   
   
  
  
    
    
    
  
        
        
       
              
               if t>1
          
               network.hidden(l).ins.v(:,:,t) =  network.hidden(l).ins.v(:,:,t) + network.hidden(l).weights.matrix*network.hidden(l).outs.v(:,:,t-1);
               else
                   network.hidden(l).ins.v(:,:,t) =  network.hidden(l).ins.v(:,:,t) + network.hidden(l).weights.matrix*network.hidden(l).outs.vp0 ;
               end
         
         if l>1
             
             network.hidden(l).ins.v(:,:,t) =  network.hidden(l).ins.v(:,:,t) + network.hidden(l).hweights.matrix*network.hidden(l-1).outs.v(:,:,t) ;
         end
               
           
    
    

      
     
               
                
       network.hidden(l).ins.v(network.hidden(l).gateind,:,t)= bsxfun(@plus,network.hidden(l).ins.v(network.hidden(l).gateind,:,t),network.hidden(l).biases.v(network.hidden(l).gateind,:)); 
       network.hidden(l).ins.v(network.hidden(l).gateind,:,t) = sigmoid(network.hidden(l).ins.v(network.hidden(l).gateind,:,t));
   

     
    network.hidden(l).ins.state(:,:,t)=network.hidden(l).ins.v(network.hidden(l).hidind,:,t).*network.hidden(l).ins.v(network.hidden(l).writeind,:,t);
 
    
            
       if t>1
            network.hidden(l).ins.state(:,:,t) = network.hidden(l).ins.state(:,:,t) + network.hidden(l).ins.state(:,:,t-1).*network.hidden(l).ins.v(network.hidden(l).keepind,:,t);
       else
           
           network.hidden(l).ins.state(:,:,t) = network.hidden(l).ins.state(:,:,t) +network.hidden(l).ins.statep0.*network.hidden(l).ins.v(network.hidden(l).keepind,:,t);
       end
    
        
    
    network.hidden(l).outs.v(:,:,t) = network.hidden(l).ins.state(:,:,t).*network.hidden(l).ins.v(network.hidden(l).readind,:,t);
    network.hidden(l).outs.v(:,:,t)=bsxfun(@plus,network.hidden(l).outs.v(:,:,t),network.hidden(l).biases.v(network.hidden(l).hidind,:)); 
     network.hidden(l).outs.v(:,:,t) = tanh(network.hidden(l).outs.v(:,:,t));
 
        network.output.outs.v(:,:,t) = network.output.outs.v(:,:,t) + network.output.weights(l).matrix*network.hidden(l).outs.v(:,:,t);
  
   

        if t==size(inputs,3)
        network.hidden(l).outs.last = network.hidden(l).outs.v(:,:,t); 
         network.hidden(l).ins.last = network.hidden(l).ins.state(:,:,t);
    end
    
    end
end

    network.output.outs.v = network.output.fx(network.output.outs.v);


end
function [network] = ForwardPasstest(network, inputs)

inputs = gpuArray(inputs);
network.input.outs.v=inputs;


   for l=1:length(network.hidden)
      hidden(l).outs.vp = network.hidden(l).outs.vp0 ;
    hidden(l).ins.statep = network.hidden(l).ins.statep0;
   end

for t=1:size(inputs,3);
    for l=1:length(network.hidden);
    

    
    
    hidden(l).ins.v= network.hidden(l).iweights.matrix*network.input.outs.v(:,:,t);
   
   
  
  
    
    
    
  
        
        
              
               
          
               hidden(l).ins.v =  hidden(l).ins.v + network.hidden(l).weights.matrix*hidden(l).outs.vp;
 
         if l>1
             
             hidden(l).ins.v =  hidden(l).ins.v + network.hidden(l).hweights.matrix*hidden(l-1).outs.v;
         end
               
           
    
    

               
                
         hidden(l).ins.v(network.hidden(l).gateind,:)= bsxfun(@plus,hidden(l).ins.v(network.hidden(l).gateind,:),network.hidden(l).biases.v(network.hidden(l).gateind,:)); 
       hidden(l).ins.v(network.hidden(l).gateind,:) = sigmoid(hidden(l).ins.v(network.hidden(l).gateind,:));
   

     
    hidden(l).ins.state=hidden(l).ins.v(network.hidden(l).hidind,:).*hidden(l).ins.v(network.hidden(l).writeind,:);
 
         ttemp = t-1;
            
       
            hidden(l).ins.state = hidden(l).ins.state + hidden(l).ins.statep.*hidden(l).ins.v(network.hidden(l).keepind,:);
     
    
        
    
    hidden(l).outs.v = hidden(l).ins.state.*hidden(l).ins.v(network.hidden(l).readind,:);
     hidden(l).outs.v=bsxfun(@plus,hidden(l).outs.v,network.hidden(l).biases.v(network.hidden(l).hidind,:)); 
     hidden(l).outs.v = tanh(hidden(l).outs.v);

        network.output.outs.v(:,:,t) = network.output.outs.v(:,:,t) + network.output.weights(l).matrix*hidden(l).outs.v;

    hidden(l).outs.vp = hidden(l).outs.v;
    hidden(l).ins.statep = hidden(l).ins.state;
  
  
        
    
    end
end

    network.output.outs.v = network.output.fx(network.output.outs.v);

    

end

function network = computegradient(network, targets,omat)
oind = find(omat);

network.output.outs.j(oind) =  network.output.outs.v(oind)- targets(oind);

network.output.outs.j = network.output.outs.j.*network.output.dx(network.output.outs.v);




for l=1:length(network.hidden)
hidden(l).ins.statej = gpuArray(zeros(network.nhidden(l),size(targets,2),'single'));
hidden(l).outs.j = gpuArray(zeros(network.nhidden(l),size(targets,2),'single'));
hidden(l).ins.j = gpuArray(zeros(network.nhidden(l)*4,size(network.input.outs.v,2),'single'));
end
for t=size(network.input.outs.v,3):-1:1;
    
 
  
   for l= length(network.hidden):-1:1
    
    network.output.weights(l).gradient = network.output.weights(l).gradient + network.output.outs.j(:,:,t)*network.hidden(l).outs.v(:,:,t)';
 
    hidden(l).outs.j = hidden(l).outs.j + network.output.weights(l).matrix'*network.output.outs.j(:,:,t) ;
   
    

   
   hidden(l).outs.j = hidden(l).outs.j.*tanhdir(network.hidden(l).outs.v(:,:,t));
   network.hidden(l).biases.j(network.hidden(l).hidind,:) = network.hidden(l).biases.j(network.hidden(l).hidind,:) + sum(hidden(l).outs.j,2);
   
   
   
 
   
   
   
   hidden(l).ins.j(network.hidden(l).readind,:) = hidden(l).outs.j.* network.hidden(l).ins.state(:,:,t) ;
  
   hidden(l).ins.statej = hidden(l).ins.statej +  hidden(l).outs.j.*network.hidden(l).ins.v(network.hidden(l).readind,:,t);

            ttemp = t-1;
            if ttemp>0
                  
                  hidden(l).ins.statejp =  hidden(l).ins.statej.*network.hidden(l).ins.v(network.hidden(l).keepind,:,t); 
                
            
           
                
            end
        
     
          
            
            if ttemp>0
                 
              
                   hidden(l).ins.j(network.hidden(l).keepind,:) =    network.hidden(l).ins.state(:,:,t-1).*hidden(l).ins.statej;
            else
                
               hidden(l).ins.j(network.hidden(l).keepind,:)= network.hidden(l).ins.statep0.*hidden(l).ins.statej;
                
            end
                
        
         hidden(l).ins.j(network.hidden(l).writeind,:) = network.hidden(l).ins.v(network.hidden(l).hidind,:,t).*hidden(l).ins.statej;
         hidden(l).ins.j(network.hidden(l).hidind,:) = network.hidden(l).ins.v(network.hidden(l).writeind,:,t).*hidden(l).ins.statej;
        
            hidden(l).ins.j(network.hidden(l).gateind,:)= hidden(l).ins.j(network.hidden(l).gateind,:).*sigdir(network.hidden(l).ins.v(network.hidden(l).gateind,:,t));
       network.hidden(l).biases.j(network.hidden(l).gateind,:) = network.hidden(l).biases.j(network.hidden(l).gateind,:) + sum(hidden(l).ins.j(network.hidden(l).gateind,:),2);
    
              

    if t-1>0
         hidden(l).outs.jp =   network.hidden(l).weights.matrix'*hidden(l).ins.j;
     
        network.hidden(l).weights.gradient = network.hidden(l).weights.gradient + hidden(l).ins.j*network.hidden(l).outs.v(:,:,t-1)';
        
       
        
    else
        
        network.hidden(l).weights.gradient = network.hidden(l).weights.gradient + hidden(l).ins.j*network.hidden(l).outs.vp0';
             
     
      end
  
    if l>1
        hidden(l-1).outs.j = hidden(l-1).outs.j + network.hidden(l).hweights.matrix'*hidden(l).ins.j;
        network.hidden(l).hweights.gradient = network.hidden(l).hweights.gradient + hidden(l).ins.j*network.hidden(l-1).outs.v(:,:,t)';
    end
  
   network.hidden(l).iweights.gradient = network.hidden(l).iweights.gradient + (hidden(l).ins.j)*network.input.outs.v(:,:,t)';
  
   
    
    
    
    
    
    hidden(l).outs.j = hidden(l).outs.jp;
    hidden(l).ins.statej = hidden(l).ins.statejp;
   end
 
   
    
end

end


function [lcost] = evalCrossEntropy(output,targets,omat)
  
 

   oind = find(omat);

    ldiff = targets.*log2(output);

  
    lcost = -1*sum(ldiff(:));
    
    
    

end

function network = updateV(network, dW) 

       ninput = network.input.n;
         noutput = network.output.n;
    
        
        start = 1;
        last = 0;
        
        for l=1:length(network.hidden)
            nhidden = network.nhidden(l);
            
            last = last + numel(network.hidden(l).iweights.matrix);
            network.hidden(l).iweights.matrix = reshape(dW(start:last),4*nhidden,ninput)+ network.hidden(l).iweights.matrix ;
            start = last + 1;

               last = last + numel(network.hidden(l).biases.v);
            network.hidden(l).biases.v = reshape(dW(start:last),4*nhidden,1)+ network.hidden(l).biases.v ;
            start = last + 1;
        
            for i=1:length(network.hidden(l).weights);
                last = last + numel(network.hidden(l).weights(i).matrix);
                network.hidden(l).weights(i).matrix = reshape(dW(start:last),4*nhidden,nhidden)+network.hidden(l).weights(i).matrix;
                start = last+1;
                if l>1
                   last = last + numel(network.hidden(l).hweights(i).matrix);
                network.hidden(l).hweights(i).matrix = reshape(dW(start:last),4*nhidden,network.nhidden(l-1))+network.hidden(l).hweights(i).matrix;
                start = last+1; 
                    
                end
   
              
                
            end
            
         
           
            
            
            
            last = last+ numel(network.output.weights(l).matrix);
            network.output.weights(l).matrix = reshape(dW(start:last),noutput,nhidden)+ network.output.weights(l).matrix ;
            start=last+1;
            
        end

end

function vect=weights2vect(allvects)
    lsum = 0;
    lengths = cell(length(allvects),1); 
    for i=1:length(allvects)
        lsum = lsum + numel(allvects{i});
        lengths{i}= lsum;
        
        
    end
    vect = zeros(lsum,1,'single');
  
    vect(1:lengths{1}) = gather(reshape(allvects{1},lengths{1},1));
    for i=2:length(allvects)
        vect(lengths{i-1}+1:lengths{i}) = gather(reshape(allvects{i},lengths{i}-lengths{i-1},1));
    end


end



function network = initpass(network,nbatch,maxt)

    ninput = network.input.n;
    
    noutput = network.output.n;
    
    
    for l=1:length(network.hidden)
       
        nhidden = network.nhidden(l);
    network.output.weights(l).gradient = gpuArray(zeros(noutput,nhidden,'single'));
    if ~network.last
        network.hidden(l).outs.vp0 =  gpuArray(zeros(nhidden,nbatch,'single'));
        network.hidden(l).ins.statep0= gpuArray(zeros(nhidden,nbatch,'single'));
    else
            network.hidden(l).outs.vp0 =   network.hidden(l).outs.last ;
          
    network.hidden(l).ins.statep0 = network.hidden(l).ins.last; 
    end
    network.hidden(l).biases.j = gpuArray(zeros(nhidden*4,1,'single'));
    for i=1:length(network.hidden(l).weights);
    network.hidden(l).weights(i).gradient = gpuArray(zeros(nhidden*4,nhidden,'single'));
 

    end
    network.hidden(l).iweights.gradient = gpuArray(zeros(nhidden*4,ninput,'single'));
    if l>1
        network.hidden(l).hweights(i).gradient = gpuArray(zeros(nhidden*4,network.nhidden(l-1),'single'));
        
    end

  
    
     network.hidden(l).outs.v = gpuArray(zeros(nhidden,nbatch,maxt,'single'));
     network.hidden(l).ins.v = gpuArray(zeros(4*nhidden,nbatch,maxt,'single'));
     network.hidden(l).ins.state = gpuArray(zeros(nhidden,nbatch,maxt,'single'));

    
     
    end
    
    
    
    network.output.outs.j = gpuArray(zeros(noutput,nbatch,maxt,'single'));
     network.input.outs.v = gpuArray(zeros(ninput,nbatch,maxt,'single'));
   
    network.output.outs.v = gpuArray(zeros(noutput,nbatch,maxt,'single'));


end

function network = initpasstest(network,nbatch,maxt)

    ninput = network.input.n;
    
    noutput = network.output.n;
    
    
    for l=1:length(network.hidden)
       
        nhidden = network.nhidden(l);
    network.output.weights(l).gradient = gpuArray(zeros(noutput,nhidden,'single'));
    if ~network.last
        network.hidden(l).outs.vp0 =  gpuArray(zeros(nhidden,nbatch,'single'));
        network.hidden(l).ins.statep0= gpuArray(zeros(nhidden,nbatch,'single'));
    else
            network.hidden(l).outs.vp0 =   network.hidden(l).outs.last ;
          
    network.hidden(l).ins.statep0 = network.hidden(l).ins.last; 
    end
    network.hidden(l).biases.j = gpuArray(zeros(nhidden*4,1,'single'));
    for i=1:length(network.hidden(l).weights);
    network.hidden(l).weights(i).gradient = gpuArray(zeros(nhidden*4,nhidden,'single'));
 

    end
    network.hidden(l).iweights.gradient = gpuArray(zeros(nhidden*4,ninput,'single'));
    if l>1
        network.hidden(l).hweights(i).gradient = gpuArray(zeros(nhidden*4,network.nhidden(l-1),'single'));
        
    end

  
    
     network.hidden(l).outs.v = gpuArray(zeros(nhidden,nbatch,length(network.storeind),'single'));
     network.hidden(l).ins.state = gpuArray(zeros(nhidden,nbatch,length(network.storeind),'single'));

    
     
    end
    
    
    
    network.output.outs.j = gpuArray(zeros(noutput,nbatch,maxt,'single'));
     network.input.outs.v = gpuArray(zeros(ninput,nbatch,maxt,'single'));
   
    network.output.outs.v = gpuArray(zeros(noutput,nbatch,maxt,'single'));


end
function network = initnetwork(ninput,nhidden,noutput)




network.input.n = ninput;
network.nhidden = nhidden;
network.output.n = noutput;









for j = 1:(length(network.nhidden))
    
    nhidden = network.nhidden(j);
   network.hidden(j).hidind = (1:nhidden)';
network.hidden(j).writeind = (nhidden+1:2*nhidden)';
network.hidden(j).keepind = (2*nhidden+1:3*nhidden)';
network.hidden(j).readind = (3*nhidden+1:4*nhidden)';
network.hidden(j).gateind = (nhidden+1:4*nhidden)';
  
network.hidden(j).iweights.matrix = gpuArray(.1*(randn(nhidden*4,ninput,'single')));  
network.hidden(j).biases.v = gpuArray(zeros(4*nhidden,1,'single'));  
network.hidden(j).biases.v(network.hidden(j).keepind)=3;
network.hidden(j).biases.v(network.hidden(j).readind)=-2;
network.hidden(j).biases.v(network.hidden(j).writeind)=0;
network.hidden(j).iweights.gated = 0;



    network.hidden(j).weights.matrix =gpuArray(.0001*(randn(nhidden*4,nhidden,'single')));

    

 
    if j>1
        network.hidden(j).hweights.matrix =gpuArray(.01*(randn(nhidden*4,network.nhidden(j-1),'single')));
    end
  
    
network.hidden(j).fx = @sigmoid;
network.hidden(j).dx = @sigdir;


network.output.weights(j).matrix = gpuArray(.1*(randn(noutput,nhidden,'single')));



end


network.nparam = length(weights2vect(getW(network)));



network.output.fx = @softmax;
network.output.dx = @softdir;
network.errorFunc = @evalCrossEntropy;
network.output.getHessian = @CrossEntropyHessian;

    
    
    

end


function J = getJ(network)
jtot=1;
J = cell(jtot,1);
c=1;
for l=1:length(network.hidden)
    J{c}= network.hidden(l).iweights.gradient;
    c=c+1;
network.hidden(l).biases.j(network.hidden(l).hidind)=0;
    J{c}=network.hidden(l).biases.j;
        c=c+1;
 
    for i = 1:length(network.hidden(l).weights);
        J{c}=network.hidden(l).weights(i).gradient;
        c=c+1;
        if l>1
            J{c}=network.hidden(l).hweights(i).gradient;
           c=c+1; 
        end
     
            
    end
       
    J{c} = network.output.weights(l).gradient;
    c=c+1;
end



end
function W = getW(network)
jtot=1; 
W = cell(jtot,1);
c=1;
for l=1:length(network.hidden)
    
     W{c}= network.hidden(l).iweights.matrix;
    c=c+1;
 
       W{c}= network.hidden(l).biases.v;
    c=c+1;
 
    
    for i = 1:length(network.hidden(l).weights);
        W{c}=network.hidden(l).weights(i).matrix;
        c=c+1;
         if l>1
           W{c}=network.hidden(l).hweights(i).matrix;
           c=c+1; 
        end
       
    end
   
    W{c} = network.output.weights(l).matrix;
    c=c+1;
end



end





function f= sigmoid(x)


f= 1./(1+ exp(-1.*x));
end

function o = softdir(x);

  o=ones(size(x),'single');


end
function o = softmax(x)
    
          o=bsxfun(@times,1./sum(exp(x),1),exp(x));
end
function dir = sigdir( y )

dir = y.*(1-y);


end
function dir = tanhdir( y )

dir = (1-y.*y);


end




%function m=gather(m)

%end
%function m=gpuArray(m)

%end