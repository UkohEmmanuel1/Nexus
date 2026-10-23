#!/usr/bin/env node
import {readFile, writeFile, mkdir, chmod, unlink, stat} from 'node:fs/promises';
import {homedir} from 'node:os';
import {join, resolve, basename} from 'node:path';
import readline from 'node:readline/promises';
import {stdin, stdout} from 'node:process';

const endpoints = {
 openai:'https://api.openai.com/v1/chat/completions',
 deepseek:'https://api.deepseek.com/chat/completions',
 groq:'https://api.groq.com/openai/v1/chat/completions',
 together:'https://api.together.xyz/v1/chat/completions',
 fireworks:'https://api.fireworks.ai/inference/v1/chat/completions',
 openrouter:'https://openrouter.ai/api/v1/chat/completions',
 mistral:'https://api.mistral.ai/v1/chat/completions',
 huggingface:'https://router.huggingface.co/v1/chat/completions',
 google:'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
 xai:'https://api.x.ai/v1/chat/completions',
 anthropic:'https://api.anthropic.com/v1/messages'
};
const examples={openai:'gpt-4.1-mini',deepseek:'deepseek-chat',groq:'llama-3.3-70b-versatile',together:'meta-llama/Llama-3.3-70B-Instruct-Turbo',fireworks:'accounts/fireworks/models/llama-v3p3-70b-instruct',openrouter:'deepseek/deepseek-chat',mistral:'mistral-small-latest',huggingface:'Qwen/Qwen2.5-72B-Instruct',google:'gemini-2.5-flash',xai:'grok-3-mini',anthropic:'claude-sonnet-4-5'};
const dir=join(homedir(),'.config','vonk');
const config=join(dir,'config.json'), secrets=join(dir,'keys.json');
async function load(p){try{return JSON.parse(await readFile(p,'utf8'))}catch{return {}}}
async function save(p,obj,secret=false){await mkdir(dir,{recursive:true,mode:0o700});await writeFile(p,JSON.stringify(obj,null,2),{mode:secret?0o600:0o644});if(secret)await chmod(p,0o600)}
function validate(p){if(!Object.hasOwn(endpoints,p))throw Error('Unknown provider: '+p)}
async function complete(provider,model,messages,key){
 const anthropic=provider==='anthropic';
 const headers=anthropic?{'x-api-key':key,'anthropic-version':'2023-06-01','content-type':'application/json'}:{Authorization:'Bearer '+key,'content-type':'application/json'};
 const system=messages.filter(m=>m.role==='system').map(m=>m.content).join('\n');
 const body=anthropic?{model,messages:messages.filter(m=>m.role!=='system'),system,max_tokens:1024}:{model,messages,max_tokens:1024};
 const response=await fetch(endpoints[provider],{method:'POST',headers,body:JSON.stringify(body),signal:AbortSignal.timeout(60000),redirect:'error'});
 if(!response.ok)throw Error('Provider returned '+response.status+'; check your model access, key and quota');
 const data=await response.json();
 return anthropic?data.content.filter(x=>x.type==='text').map(x=>x.text).join(''):data.choices[0].message.content;
}
async function main(){
 const [command='chat',...args]=process.argv.slice(2);
 if(command==='models'){for(const p of Object.keys(endpoints))console.log(p+' : '+examples[p]+' (example)');return}
 if(command==='use'){const [p,model]=args;validate(p);if(!model)throw Error('Usage: nexus-ai use PROVIDER MODEL');await save(config,{provider:p,model});console.log('Selected '+p+'/'+model);return}
 if(command==='logout'){await unlink(secrets).catch(()=>{});console.log('Stored API keys deleted');return}
 if(command==='login'){
  const [p]=args;validate(p);const rl=readline.createInterface({input:stdin,output:stdout});
  try{console.log('Warning: key entry is visible while typing. Prefer environment variables on shared terminals.');const key=(await rl.question(p+' API key: ')).trim();if(!key)throw Error('Empty key');await save(secrets,{...await load(secrets),[p]:key},true);console.log('Saved locally')}finally{rl.close()}return
 }
 if(command!=='chat')throw Error('Commands: chat, models, use, login, logout');
 const settings={provider:'deepseek',model:'deepseek-chat',...await load(config)};
 const p=settings.provider;validate(p);
 const key=process.env[p.toUpperCase()+'_API_KEY']||(await load(secrets))[p];
 if(!key)throw Error('No provider key. Set '+p.toUpperCase()+'_API_KEY or run nexus-ai login '+p);
 let prompt=[],files=[];
 for(let i=0;i<args.length;i++){if(args[i]==='--file'){files.push(args[++i]);}else prompt.push(args[i])}
 const context=[];
 for(const path of files){const full=resolve(path);if((await stat(full)).size>60000)throw Error('File over 60KB: '+path);context.push('File '+basename(full)+'\n'+await readFile(full,'utf8'))}
 let messages=[{role:'system',content:'You are Vonk, a practical coding assistant. Never claim you edited local files.\n'+context.join('\n')}];
 async function ask(q){messages.push({role:'user',content:q});try{const response=await complete(p,settings.model,messages,key);console.log('\nVonk: '+response+'\n');messages.push({role:'assistant',content:response})}catch(e){messages.pop();console.error(e.message)}}
 if(prompt.length){await ask(prompt.join(' '));return}
 const rl=readline.createInterface({input:stdin,output:stdout});
 console.log('Vonk AI | '+p+'/'+settings.model+' | /exit /clear');
 try{while(true){const q=(await rl.question('nexus> ')).trim();if(['/exit','/quit'].includes(q))break;if(q==='/clear'){messages=messages.slice(0,1);continue}if(q)await ask(q)}}catch(e){if(e.code!=='ERR_USE_AFTER_CLOSE')throw e}finally{rl.close()}
}
main().catch(e=>{console.error('Vonk:',e.message);process.exitCode=1});
