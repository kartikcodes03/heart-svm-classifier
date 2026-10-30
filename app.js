// Browser-side inference using support vectors exported from the user's trained scikit-learn SVM pipelines.
let modelData;
const $ = (id) => document.getElementById(id);
fetch('model-data.json').then(r => { if(!r.ok) throw new Error('Model data could not be loaded.'); return r.json(); }).then(data => { modelData=data; }).catch(err => { $('empty').innerHTML='<b>Model file error</b><span>'+err.message+' Keep model-data.json in the same folder as index.html.</span>'; });

// Convert the entered raw values into the exact scaled + one-hot feature vector used in training.
function transformInput(raw){
  const vector=[];
  modelData.numericColumns.forEach((name,i)=>vector.push((Number(raw[name])-modelData.means[i])/modelData.scales[i]));
  modelData.categoricalColumns.forEach((name,i)=>{
    modelData.categories[i].forEach(category=>vector.push(raw[name]===category?1:0));
  });
  return vector;
}

// Calculate the kernel value between the new sample and one support vector.
function kernelValue(x,sv,params){
  let dot=0,dist=0;
  for(let i=0;i<x.length;i++){dot+=x[i]*sv[i];const d=x[i]-sv[i];dist+=d*d;}
  if(params.kernel==='linear') return dot;
  if(params.kernel==='poly') return Math.pow(params.gamma*dot+params.coef0,params.degree);
  return Math.exp(-params.gamma*dist);
}

// Reproduce the binary SVM decision function from the saved support vectors.
function predictKernel(x,params){
  let decision=params.intercept;
  for(let i=0;i<params.supportVectors.length;i++){
    decision+=params.dualCoef[i]*kernelValue(x,params.supportVectors[i],params);
  }
  return decision>=0?1:0;
}

$('form').addEventListener('submit',event=>{
  event.preventDefault();
  if(!modelData){alert('The trained model data is still loading. Please try again.');return;}
  const formData=new FormData(event.currentTarget);
  const raw=Object.fromEntries(formData.entries());
  const x=transformInput(raw);
  const labels={0:'No heart disease detected by model',1:'Heart disease detected by model'};
  $('cards').innerHTML='';
  ['linear','poly','rbf'].forEach(kernel=>{
    const prediction=predictKernel(x,modelData.models[kernel]);
    const card=document.createElement('div');card.className='card';
    const title=document.createElement('div');
    const name=document.createElement('b');name.textContent=kernel+' kernel';
    const detail=document.createElement('small');detail.textContent='SVM classification output';
    title.append(name,detail);
    const result=document.createElement('span');result.className='prediction '+(prediction===1?'positive':'negative');result.textContent=labels[prediction];
    card.append(title,result);$('cards').append(card);
  });
  $('empty').classList.add('hidden');$('cards').classList.remove('hidden');
});
