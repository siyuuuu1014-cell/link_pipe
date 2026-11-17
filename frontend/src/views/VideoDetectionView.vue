<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api, absoluteAsset } from '../api'
import DisclaimerBlock from '../components/DisclaimerBlock.vue'

const file = ref<File | null>(null)
const preview = ref('')
const confidence = ref(0.25)
const iou = ref(0.5)
const frameInterval = ref(1)
const loading = ref(false)
const result = ref<any>(null)
function selectFile(upload: any) { file.value = upload.raw; preview.value = URL.createObjectURL(upload.raw); result.value = null }
async function run() {
  if (!file.value) return ElMessage.warning('请选择 MP4 视频')
  const form = new FormData(); form.append('file', file.value); form.append('confidence', String(confidence.value)); form.append('iou', String(iou.value)); form.append('frame_interval', String(frameInterval.value))
  loading.value = true
  try { result.value = (await api.post('/detect/video', form)).data }
  catch (error: any) { ElMessage.error(error.response?.data?.detail || '视频处理失败') }
  finally { loading.value = false }
}
</script>

<template>
  <div class="page-header"><div><h2>视频逐帧检测</h2><p>上传 MP4，生成带框结果视频和疑似泄漏时间区间</p></div></div>
  <div class="card"><el-upload drag :auto-upload="false" :limit="1" accept="video/mp4" :on-change="selectFile"><div class="upload-panel"><strong>拖拽 MP4 视频到此处</strong><p class="muted">最大 500 MB，演示时建议使用短视频</p></div></el-upload>
    <div class="control-row"><label>置信度 {{ confidence.toFixed(2) }}<el-slider v-model="confidence" :min="0.05" :max="0.95" :step="0.05" /></label><label>IoU {{ iou.toFixed(2) }}<el-slider v-model="iou" :min="0.1" :max="0.9" :step="0.05" /></label><div><el-input-number v-model="frameInterval" :min="1" :max="120" /><el-button type="primary" size="large" :loading="loading" @click="run">开始处理</el-button></div></div>
  </div>
  <template v-if="result"><div class="spacer"></div><div class="status-banner"><strong>{{ result.summary }}</strong><span>时长 {{ result.duration.toFixed(1) }} 秒 · {{ result.frame_count }} 帧</span></div><div class="spacer"></div>
    <div class="grid-2"><div class="card"><h3>原始视频</h3><video class="video-frame" :src="preview" controls /></div><div class="card"><h3>结果视频</h3><video class="video-frame" :src="absoluteAsset(result.result_video_url)" controls /><a :href="absoluteAsset(result.result_video_url)" download><el-button>下载结果视频</el-button></a></div></div>
    <div class="spacer"></div><div class="grid-2"><div class="card"><h3>目标统计</h3><el-descriptions :column="1" border><el-descriptions-item label="疑似泄漏">{{ result.counts.leak }}</el-descriptions-item><el-descriptions-item label="裂纹">{{ result.counts.crack }}</el-descriptions-item><el-descriptions-item label="可见水体">{{ result.counts.water }}</el-descriptions-item></el-descriptions></div><div class="card"><h3>疑似泄漏时间区间</h3><el-tag v-for="range in result.leak_time_ranges" :key="range.join('-')" type="danger" style="margin:4px">{{ range[0] }}s – {{ range[1] }}s</el-tag><div v-if="!result.leak_time_ranges.length" class="empty-state">未检测到疑似泄漏时间段</div></div></div>
    <div class="spacer"></div><DisclaimerBlock />
  </template>
</template>
