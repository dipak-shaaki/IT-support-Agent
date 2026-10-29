const chat = document.getElementById('chat');
const welcomeTemplate = document.getElementById('welcomeTemplate');
const form = document.getElementById('chatForm');
const question = document.getElementById('question');
const trace = document.getElementById('trace');
const sourceUsed = document.getElementById('sourceUsed');
const newChatButton = document.getElementById('newChat');
const conversationStorageKey = 'enterprise-it-support-conversation-id';
const adminKeyStorageKey = 'it_copilot_admin_key';
let conversationId = localStorage.getItem(conversationStorageKey);
let restorePromise = Promise.resolve();

function showWelcome(){
  const content=welcomeTemplate.content.cloneNode(true);
  const welcome=content.querySelector('.simple-welcome');
  welcome.querySelector('.simple-welcome-composer').append(form,document.querySelector('.simple-hint'));
  chat.replaceChildren(content);
  document.querySelector('.simple-shell').classList.add('is-empty');
}
function removeWelcome(){
  const welcome=chat.querySelector('.simple-welcome');
  if(!welcome) return;
  const composeArea=document.querySelector('.simple-compose-area');
  composeArea.prepend(form);
  composeArea.append(document.querySelector('.simple-hint'));
  welcome.remove();
  document.querySelector('.simple-shell').classList.remove('is-empty');
}
function escapeHtml(s=''){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
function formatText(s=''){return escapeHtml(s).replace(/\n/g,'<br>');}
function messageHtml(text, source='', citations=[]){
  const citeHtml=citations.length?`<div class="citations"><strong>Sources</strong><br>${citations.map(c=>c.url?`<a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.title)}</a>`:escapeHtml(c.title)).join('<br>')}</div>`:'';
  return `${formatText(text)}${source?`<div class="answer-source">Source: ${escapeHtml(source)}</div>`:''}${citeHtml}`;
}
function addMessage(role, text, source='', citations=[]){
  removeWelcome();
  const wrap=document.createElement('div'); wrap.className=`message ${role}`;
  wrap.innerHTML=`<div class="avatar">AI</div><div class="bubble">${messageHtml(text, source, citations)}</div>`;
  chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
}
function addStreamingBubble(){
  removeWelcome();
  const wrap=document.createElement('div'); wrap.className='message assistant';
  wrap.innerHTML='<div class="avatar">AI</div><div class="bubble"><span class="thinking">Thinking…</span></div>';
  chat.appendChild(wrap); chat.scrollTop=chat.scrollHeight;
  return wrap.querySelector('.bubble');
}
function renderTrace(items=[]){trace.innerHTML=items.length?items.map(x=>`<div class="trace-item">${escapeHtml(x)}</div>`).join(''):'<div class="empty">No trace.</div>';}
function appendTrace(text){
  const empty=trace.querySelector('.empty'); if(empty) empty.remove();
  const item=document.createElement('div'); item.className='trace-item'; item.textContent=text;
  trace.appendChild(item); trace.scrollTop=trace.scrollHeight;
}
async function askAgent(q){
  await restorePromise;
  addMessage('user',q); question.value=''; trace.innerHTML='<div class="empty">Connecting…</div>'; sourceUsed.textContent='Running';
  const btn=form.querySelector('button'); btn.disabled=true; newChatButton.disabled=true;
  const bubble=addStreamingBubble();
  let streamedAnswer='';
  try{
    const body={question:q}; if(conversationId) body.conversation_id=conversationId;
    const res=await fetch('/api/chat/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!res.ok) throw new Error(`Request failed (${res.status})`);
    if(!res.body) throw new Error('Streaming is not supported by this browser');
    const reader=res.body.getReader(); const decoder=new TextDecoder(); let buf=''; let finished=false;
    while(true){
      const {value, done}=await reader.read();
      if(done) break;
      buf+=decoder.decode(value,{stream:true});
      const chunks=buf.split('\n\n'); buf=chunks.pop();
      for(const chunk of chunks){
        const line=chunk.split('\n').find(l=>l.startsWith('data:'));
        if(!line) continue;
        const evt=JSON.parse(line.slice(5).trim());
        if(evt.type==='trace') appendTrace(evt.text);
        else if(evt.type==='token'){
          streamedAnswer+=evt.text;
          bubble.innerHTML=formatText(streamedAnswer);
          chat.scrollTop=chat.scrollHeight;
        }
        else if(evt.type==='done'){
          finished=true;
          conversationId=evt.conversation_id;
          localStorage.setItem(conversationStorageKey,conversationId);
          bubble.innerHTML=messageHtml(evt.answer, evt.source_used, evt.citations||[]);
          sourceUsed.textContent=evt.source_used;
        } else if(evt.type==='error') throw new Error(evt.message);
      }
    }
    if(!finished) throw new Error('Stream ended before an answer');
  }catch(e){
    bubble.innerHTML=messageHtml(`Error: ${e.message}`);
    appendTrace('Request failed');
    sourceUsed.textContent='Error';
  }finally{
    btn.disabled=false; newChatButton.disabled=false; chat.scrollTop=chat.scrollHeight;
  }
}
form.addEventListener('submit',e=>{e.preventDefault(); const q=question.value.trim(); if(q) askAgent(q);});
question.addEventListener('keydown',event=>{
  if(event.key==='Enter'&&!event.shiftKey){
    event.preventDefault();
    form.requestSubmit();
  }
});
question.addEventListener('input',()=>{
  question.style.height='auto';
  question.style.height=`${Math.min(question.scrollHeight,160)}px`;
});
chat.addEventListener('click',event=>{
  const example=event.target.closest('.example');
  if(example) askAgent(example.textContent.trim());
});

newChatButton.addEventListener('click',async()=>{
  await restorePromise;
  conversationId=null;
  localStorage.removeItem(conversationStorageKey);
  showWelcome();
  renderTrace();
  sourceUsed.textContent='Ready';
});

async function restoreConversation(){
  if(!conversationId) return;
  try{
    const response=await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`);
    if(!response.ok) throw new Error(`Could not restore conversation (${response.status})`);
    const data=await response.json();
    if(data.messages?.length) chat.replaceChildren();
    for(const message of data.messages||[]) addMessage(message.role==='user'?'user':'assistant',message.content);
  }catch(error){
    appendTrace(`Conversation history could not be restored: ${error.message}`);
    sourceUsed.textContent='Memory unavailable';
  }
}
showWelcome();
restorePromise=restoreConversation();

const uploadForm=document.getElementById('uploadForm');
const uploadButton=document.getElementById('uploadBtn');
const uploadStatus=document.getElementById('uploadStatus');
const adminKeyInput=document.getElementById('adminKey');
const savedAdminKey=sessionStorage.getItem(adminKeyStorageKey);
if(savedAdminKey) adminKeyInput.value=savedAdminKey;

uploadForm.addEventListener('submit',async event=>{
  event.preventDefault();
  const file=document.getElementById('fileInput').files[0];
  const key=adminKeyInput.value.trim();
  if(!file){uploadStatus.textContent='Choose a document to upload.'; return;}
  if(!key){uploadStatus.textContent='Enter the admin API key.'; return;}
  const formData=new FormData();
  formData.append('file',file);
  uploadButton.disabled=true;
  uploadStatus.textContent='Uploading and indexing…';
  sessionStorage.setItem(adminKeyStorageKey,key);
  try{
    const response=await fetch('/api/ingest',{
      method:'POST',
      headers:{'X-Admin-Key':key},
      body:formData
    });
    const data=await response.json();
    if(!response.ok) throw new Error(data.detail||`Upload failed (${response.status})`);
    uploadStatus.textContent=`Indexed ${data.file}: ${data.chunks} chunks added.`;
    uploadForm.reset();
    adminKeyInput.value=key;
  }catch(error){
    uploadStatus.textContent=`Error: ${error.message}`;
  }finally{
    uploadButton.disabled=false;
  }
});
