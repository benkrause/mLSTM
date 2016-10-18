function mLSTMeval



gpuDevice(1)

sequence = processtextfile('enwik8');

weightsfname = 'mLSTMhutter.mat';

nunits = 205;







network = initnetwork(nunits,[1900],nunits);

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
             network.output.outs.v = .999999*network.output.outs.v + .000001*(ones(size(network.output.outs.v)))/size(network.output.outs.v,1);
           [terr]=network.errorFunc(network.output.outs.v,targets,oind);
            errsum = errsum + terr;
          
            errcount = errcount+1;
            
            
        
        

      
        
        
       err = errsum/errcount;
 
       
            
    


end

function network = ForwardPass(network, inputs)

inputs = gpuArray(inputs);
network.input.outs.v = inputs;


for t=1:size(inputs,3);
  l=1;
    
    
  
    network.hidden(l).ins.v(:,:,t) =  network.hidden(l).ins.v(:,:,t) + network.hidden(l).iweights.matrix*network.input.outs.v(:,:,t);

         if t-1>0
              
               
                network.hidden(l).intermediates.v(:,:,t) =  network.hidden(l).mweights.matrix*network.hidden(l).outs.v(:,:,t-1);
                
         else
             
             network.hidden(l).intermediates.v(:,:,t) =  network.hidden(l).mweights.matrix*network.hidden(l).outs.vp0;
         end
                
                network.hidden(l).factor.v(:,:,t) = network.hidden(l).fweights.matrix*inputs(:,:,t);
                network.hidden(l).mult.v(:,:,t) =   network.hidden(l).factor.v(:,:,t).*network.hidden(l).intermediates.v(:,:,t);
          
   
               
              
         
              
               
                network.hidden(l).ins.v(:,:,t) =  network.hidden(l).ins.v(:,:,t) + network.hidden(l).weights.matrix*network.hidden(l).mult.v(:,:,t);
            network.hidden(l).ins.v(network.gateind,:,t)= bsxfun(@plus,network.hidden(l).ins.v(network.gateind,:,t),network.hidden(l).biases.v(network.gateind,:));      
       network.hidden(l).ins.v(network.gateind,:,t) = sigmoid(network.hidden(l).ins.v(network.gateind,:,t));
                
                
   

    
    network.hidden(l).ins.state(:,:,t)= network.hidden(l).ins.v(network.hidind,:,t).*network.hidden(l).ins.v(network.writeind,:,t);
 
       
            
            if t-1>0
        network.hidden(l).ins.state(:,:,t) = network.hidden(l).ins.state(:,:,t)+network.hidden(l).ins.state(:,:,t-1).*network.hidden(l).ins.v(network.keepind,:,t);
            else
    network.hidden(l).ins.state(:,:,t) =network.hidden(l).ins.state(:,:,t)+ network.hidden(l).ins.statep0.*network.hidden(l).ins.v(network.keepind,:,t);;
            end
        
    
    network.hidden(l).outs.v(:,:,t) = network.hidden(l).ins.state(:,:,t).*network.hidden(l).ins.v(network.readind,:,t);;
    network.hidden(l).outs.v(:,:,t)=bsxfun(@plus,network.hidden(l).outs.v(:,:,t),network.hidden(l).biases.v(network.hidind,:));  
     network.hidden(l).outs.v(:,:,t) = tanh(network.hidden(l).outs.v(:,:,t));
     

     
     
    network.output.outs.v(:,:,t) = network.output.outs.v(:,:,t) + network.output.weights(l).matrix*network.hidden(l).outs.v(:,:,t);


      if t==size(inputs,3)
        network.hidden(l).outs.last = network.hidden(l).outs.v(:,:,t);
         network.hidden(l).ins.last =  network.hidden(l).ins.state(:,:,t);
    end

    
end
  
    network.output.outs.v = network.output.fx(network.output.outs.v);
 
    

end
function network = ForwardPasstest(network, inputs)

inputs = gpuArray(inputs);
network.input.outs.v = inputs;



    hidden.outs.vp = gpuArray(zeros(network.nhidden,size(inputs,2),'single'));
    hidden.ins.statep = gpuArray(zeros(network.nhidden,size(inputs,2),'single'));
    hidden.mult.v=gpuArray(zeros(network.nhidden,size(inputs,2),'single'));
    
   for l=1:length(network.hidden)
      hidden(l).outs.vp = network.hidden(l).outs.vp0 ;
    hidden(l).ins.statep = network.hidden(l).ins.statep0;
   end
    s=1;
for t=1:size(inputs,3);
    l=1;
  
    
    
    
    hidden.ins.v= network.hidden(l).iweights.matrix*network.input.outs.v(:,:,t);
   
   
  
  
    
    
    
  
        
 
              
               
               hidden.intermediates.v =  network.hidden(l).mweights.matrix*hidden(l).outs.vp;
               hidden.factor.v = network.hidden(l).fweights.matrix*inputs(:,:,t);
               hidden.mult.v =  hidden(l).factor.v.*hidden(l).intermediates.v;
      
   
               
           
    
    

       
     
               
                hidden.ins.v =  hidden(l).ins.v + network.hidden.weights.matrix*hidden(l).mult.v;
          hidden(l).ins.v(network.gateind,:)= bsxfun(@plus,hidden(l).ins.v(network.gateind,:),network.hidden(l).biases.v(network.gateind,:)); 
       hidden.ins.v(network.gateind,:) = sigmoid(hidden.ins.v(network.gateind,:));
   

     
    hidden.ins.state=hidden.ins.v(network.hidind,:).*hidden.ins.v(network.writeind,:);
 
   
            
    
            hidden.ins.state = hidden.ins.state + hidden(l).ins.statep.*hidden.ins.v(network.keepind,:);
      
    
        
    
    hidden.outs.v = hidden(l).ins.state.*hidden.ins.v(network.readind,:);
     hidden(l).outs.v=bsxfun(@plus,hidden(l).outs.v,network.hidden(l).biases.v(network.hidind,:)); 
     hidden.outs.v = tanh(hidden(l).outs.v);

        network.output.outs.v(:,:,t) = network.output.outs.v(:,:,t) + network.output.weights(l).matrix*hidden(l).outs.v;

    hidden.outs.vp = hidden(l).outs.v;
    hidden.ins.statep = hidden(l).ins.state;
    

    
end


    network.output.outs.v = network.output.fx(network.output.outs.v);

    

end




function network = computegradient(network, targets,omat)
oind = find(omat);

network.output.outs.j(oind) =  network.output.outs.v(oind)- targets(oind);

network.output.outs.j = network.output.outs.j.*network.output.dx(network.output.outs.v);





hidden.ins.statej = gpuArray(zeros(network.nhidden,size(targets,2),'single'));
hidden.outs.j = gpuArray(zeros(network.nhidden,size(targets,2),'single'));
hidden.ins.j = gpuArray(zeros(network.nhidden*4,size(network.input.outs.v,2),'single'));
for t=size(network.input.outs.v,3):-1:1;
  
 
  
    l=1;
 
    network.output.weights(l).gradient = network.output.weights(l).gradient + network.output.outs.j(:,:,t)*network.hidden(l).outs.v(:,:,t)';
 
    hidden.outs.j = hidden.outs.j + network.output.weights(l).matrix'*network.output.outs.j(:,:,t) ;
   
    
  
    
    

   
   hidden(l).outs.j = hidden(l).outs.j.*tanhdir(network.hidden(l).outs.v(:,:,t));
     network.hidden(l).biases.j(network.hidind,:) = network.hidden(l).biases.j(network.hidind,:) + sum(hidden(l).outs.j,2);%biases only have 1 dimension
   
   
   
 
   
   
   
   hidden(l).ins.j(network.readind,:) = hidden(l).outs.j.* network.hidden(l).ins.state(:,:,t) ;
  
   hidden(l).ins.statej = hidden(l).ins.statej +  hidden(l).outs.j.*network.hidden(l).ins.v(network.readind,:,t);

   
       
         
            ttemp = t-1;
            if ttemp>0
                  
                  hidden(l).ins.statejp =  hidden(l).ins.statej.*network.hidden(l).ins.v(network.keepind,:,t); 
               
                
            end
        
     
          
            
            if ttemp>0
                 
                 
                   hidden(l).ins.j(network.keepind,:) =    network.hidden(l).ins.state(:,:,t-1).*hidden(l).ins.statej;
            else
                
               
                 hidden(l).ins.j(network.keepind,:)= network.hidden(l).ins.statep0.*hidden(l).ins.statej;
            end
                
        
         hidden(l).ins.j(network.writeind,:) = network.hidden(l).ins.v(network.hidind,:,t).*hidden(l).ins.statej;
         hidden(l).ins.j(network.hidind,:) = network.hidden(l).ins.v(network.writeind,:,t).*hidden(l).ins.statej;
        
            hidden(l).ins.j(network.gateind,:)= hidden(l).ins.j(network.gateind,:).*sigdir(network.hidden(l).ins.v(network.gateind,:,t));
        network.hidden(l).biases.j(network.gateind,:) = network.hidden(l).biases.j(network.gateind,:) + sum(hidden(l).ins.j(network.gateind,:),2);
    
             
               
                 hidden(l).mult.j =    network.hidden(l).weights.matrix'*hidden(l).ins.j;
             

         
                 network.hidden(l).weights.gradient = network.hidden(l).weights.gradient + (hidden(l).ins.j)*network.hidden(l).mult.v(:,:,t)';
                
        
       

     if t-1>0
        
        hidden(l).intermediates.j = hidden(l).mult.j.*network.hidden(l).factor.v(:,:,t);
        
        hidden(l).factor.j= hidden(l).mult.j.*network.hidden(l).intermediates.v(:,:,t);
        
        network.hidden(l).fweights.gradient = network.hidden(l).fweights.gradient + hidden(l).factor.j*network.input.outs.v(:,:,t)';
        
        hidden(l).outs.jp =   network.hidden(l).mweights.matrix'*hidden(l).intermediates.j;
        network.hidden(l).mweights.gradient = network.hidden(l).mweights.gradient + hidden(l).intermediates.j*network.hidden(l).outs.v(:,:,t-1)';
     else
            hidden(l).intermediates.j = hidden(l).mult.j.*network.hidden(l).factor.v(:,:,t);
        
        hidden(l).factor.j= hidden(l).mult.j.*network.hidden(l).intermediates.v(:,:,t);
        
        network.hidden(l).fweights.gradient = network.hidden(l).fweights.gradient + hidden(l).factor.j*network.input.outs.v(:,:,t)';
        
        hidden(l).outs.jp =   network.hidden(l).mweights.matrix'*hidden(l).intermediates.j;
        network.hidden(l).mweights.gradient = network.hidden(l).mweights.gradient + hidden(l).intermediates.j*network.hidden(l).outs.vp0';     
         
         
         
         
     end
   

   network.hidden(l).iweights.gradient = network.hidden(l).iweights.gradient + (hidden(l).ins.j)*network.input.outs.v(:,:,t)';
  
   
    
    
    
    
    
    hidden(l).outs.j = hidden(l).outs.jp;
    hidden(l).ins.statej = hidden(l).ins.statejp;

   
    
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
            
            
               last = last + numel(network.hidden(l).fweights.matrix);
            network.hidden(l).fweights.matrix = reshape(dW(start:last),nhidden,ninput)+ network.hidden(l).fweights.matrix ;
            start = last + 1;
        
        
            for i=1:length(network.hidden(l).weights);
                last = last + numel(network.hidden(l).weights(i).matrix);
                network.hidden(l).weights(i).matrix = reshape(dW(start:last),4*nhidden,nhidden)+network.hidden(l).weights(i).matrix;
                start = last+1;
                  last = last + numel(network.hidden(l).mweights(i).matrix);
                network.hidden(l).mweights(i).matrix = reshape(dW(start:last),nhidden,nhidden)+network.hidden(l).mweights(i).matrix;
                start = last+1;
              
                
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
    
    network.hidden(l).biases.j = gpuArray(zeros(nhidden*4,1,'single'));
    if ~network.last
        network.hidden(l).outs.vp0 =  gpuArray(zeros(nhidden,nbatch,'single'));
        network.hidden(l).ins.statep0= gpuArray(zeros(nhidden,nbatch,'single'));
    else
            network.hidden(l).outs.vp0 =   network.hidden(l).outs.last ;
          
    network.hidden(l).ins.statep0 = network.hidden(l).ins.last; 
    end
      network.hidden(l).outs.v = gpuArray(zeros(nhidden,nbatch,maxt,'single'));
   

     network.hidden(l).ins.v = gpuArray(zeros(nhidden*4,nbatch,maxt,'single'));
    
      
     network.hidden(l).intermediates.v = gpuArray(zeros(nhidden,nbatch,maxt,'single'));
     
     network.hidden(l).factor.v = gpuArray(zeros(nhidden,nbatch,maxt,'single'));
     
     network.hidden(l).mult.v = gpuArray(zeros(nhidden,nbatch,maxt,'single'));
     
     
     network.hidden(l).ins.state = gpuArray(zeros(nhidden,nbatch,maxt,'single'));
     
    
    
    

    for i=1:length(network.hidden(l).weights);
    network.hidden(l).weights(i).gradient = gpuArray(zeros(nhidden*4,nhidden,'single'));
    network.hidden(l).mweights(i).gradient = gpuArray(zeros(nhidden,nhidden,'single'));

    end
    network.hidden(l).iweights.gradient = gpuArray(zeros(nhidden*4,ninput,'single'));
    network.hidden(l).fweights.gradient = gpuArray(zeros(nhidden,ninput,'single'));

  
    
     network.hidden(l).outs.v = gpuArray(zeros(nhidden,nbatch,maxt,'single'));
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
   
    
    network.hidden(l).biases.j = gpuArray(zeros(nhidden*4,1,'single'));
    if ~network.last
        network.hidden(l).outs.vp0 =  gpuArray(zeros(nhidden,nbatch,'single'));
        network.hidden(l).ins.statep0= gpuArray(zeros(nhidden,nbatch,'single'));
    else
            network.hidden(l).outs.vp0 =   network.hidden(l).outs.last ;
          
    network.hidden(l).ins.statep0 = network.hidden(l).ins.last; 
    end
  

  
    
     network.hidden(l).outs.v = gpuArray(zeros(nhidden,nbatch,'single'));
     network.hidden(l).ins.state = gpuArray(zeros(nhidden,nbatch,'single'));

    
     
    end
    
    
    
    network.output.outs.j = gpuArray(zeros(noutput,nbatch,maxt,'single'));
     network.input.outs.v = gpuArray(zeros(ninput,nbatch,maxt,'single'));
   
    network.output.outs.v = gpuArray(zeros(noutput,nbatch,maxt,'single'));


end


function network = initnetwork(ninput,nhidden,noutput)


network.input.n = ninput;
network.nhidden = nhidden;
network.output.n = noutput;




network.hidind = (1:nhidden)';
network.writeind = (nhidden+1:2*nhidden)';
network.keepind = (2*nhidden+1:3*nhidden)';
network.readind = (3*nhidden+1:4*nhidden)';
network.gateind = (nhidden+1:4*nhidden)';
for j = 1:1
    nhidden = network.nhidden(j);
   
  
network.hidden(j).iweights.matrix = gpuArray(.1*(randn(nhidden*4,ninput,'single')));  
network.hidden(j).fweights.matrix = gpuArray(.1*(randn(nhidden,ninput,'single'))); 
network.hidden(j).iweights.gated = 0;


network.hidden(j).biases.v = gpuArray(zeros(4*nhidden,1,'single'));  
network.hidden(j).biases.v(network.keepind)=3;


    network.hidden(j).weights.matrix =gpuArray(.02*(randn(nhidden*4,nhidden,'single')));
    network.hidden(j).mweights.matrix =gpuArray(.02*(randn(nhidden,nhidden,'single')));
 
  
    

network.hidden(j).fx = @sigmoid;
network.hidden(j).dx = @sigdir;
network.output.weights(j).matrix = gpuArray(.1*(randn(noutput,nhidden,'single')));
network.output.weights(j).utime = 0;
network.output.weights(j).index = j;
network.output.weights(j).gated = 0;


end


network.nparam = length(weights2vect(getW(network)));


network.output.fx = @softmax;
network.output.dx = @softdirXent;
network.errorFunc = @evalCrossEntropy;

end


function J = getJ(network)
jtot=1;
J = cell(jtot,1);
c=1;
for l=1:length(network.hidden)
    J{c}= network.hidden(l).iweights.gradient;
    c=c+1;
     J{c}= network.hidden(l).biases.j;
    c=c+1;
     J{c}= network.hidden(l).fweights.gradient;
    c=c+1;
   
 
    for i = 1:length(network.hidden(l).weights);
        J{c}=network.hidden(l).weights(i).gradient;
        c=c+1;
              J{c}=network.hidden(l).mweights(i).gradient;
        c=c+1;
     
            
    end
       
    J{c} = 1*network.output.weights(l).gradient;
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
     W{c}= network.hidden(l).fweights.matrix;
    c=c+1;
  
 
    
    for i = 1:length(network.hidden(l).weights);
        W{c}=network.hidden(l).weights(i).matrix;
        c=c+1;
           W{c}=network.hidden(l).mweights(i).matrix;
           c=c+1;
       
    end
   
    W{c} = network.output.weights(l).matrix;
    c=c+1;
end



end





function f= sigmoid(x)


f= 1./(1+ exp(-1.*x));
end

function o = softdirXent(x);

  o=ones(size(x),'single');


end

function dir = sigdir( y )

dir = y.*(1-y);


end
function dir = tanhdir( y )

dir = (1-y.*y);


end
function o = softmax(x)
    
          o=bsxfun(@times,1./sum(exp(x),1),exp(x));
end

%function m=gather(m)

%end
%function m=gpuArray(m)

%end