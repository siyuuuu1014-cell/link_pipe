<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { api, absoluteAsset } from '../api'
import RiskTag from '../components/RiskTag.vue'
import DisclaimerBlock from '../components/DisclaimerBlock.vue'
import type { DetectionResult } from '../types'

const file = ref<File | null>(null)
const preview = ref('')
const confidence = ref(0.25)
const iou = ref(0.5)
const loading = ref(false)
const result = ref<DetectionResult | null>(null)
const uploadKey = ref(0)

function selectFile(upload: any) {
  if (!upload.raw) return
  if (preview.value) URL.revokeObjectURL(preview.value)
  file.value = upload.raw
  preview.value = URL.createObjectURL(upload.raw)
  result.value = null
  uploadKey.value += 1
}
async function runDetection() {
  if (!file.value) return ElMessage.warning('请先选择图片')
  const form = new FormData()
  form.append('file', file.value)
  form.append('confidence', String(confidence.value))
  form.append('iou', String(iou.value))
  loading.value = true
  try {
    result.value = (await api.post('/detect/image', form)).data
  } catch (error: any) {
    ElMessage.error(error.response?.data?.detail || '检测失败，请检查后端和模型状态')
  } finally { loading.value = false }
}
</script>

<template>
  <div class="page-header"><div><h2>图片智能检测</h2><p>支持 JPG、JPEG、PNG、WEBP，自动定位多个异常目标</p></div></div>
  <div class="card">
    <el-upload :key="uploadKey" drag :auto-upload="false" :show-file-list="false" accept="image/jpeg,image/png,image/webp" :on-change="selectFile">
      <div class="upload-panel"><strong>拖拽管道图片到此处，或点击选择</strong><p class="muted">{{ file ? `已选择：${file.name}` : '单张图片最大 20 MB' }}</p></div>
    </el-upload>
    <div class="control-row">
      <label>置信度阈值 {{ confidence.toFixed(2) }}<el-slider v-model="confidence" :min="0.05" :max="0.95" :step="0.05" /></label>
      <label>IoU 阈值 {{ iou.toFixed(2) }}<el-slider v-model="iou" :min="0.1" :max="0.9" :step="0.05" /></label>
      <div><el-button type="primary" size="large" :loading="loading" @click="runDetection">开始检测</el-button></div>
    </div>
  </div>
  <template v-if="result">
    <div class="spacer"></div>
    <div class="status-banner"><div><strong>{{ result.status_zh }}</strong><p style="margin:5px 0 0">共检测到 {{ result.detections.length }} 个目标，推理 {{ result.inference_ms.toFixed(1) }} ms</p></div><RiskTag :level="result.risk_level" /></div>
    <div class="spacer"></div>
    <div class="image-compare">
      <div class="image-frame"><span class="image-label">原始图片</span><img :src="preview || absoluteAsset(result.original_image_url)" alt="原始管道图片"></div>
      <div class="image-frame"><span class="image-label">检测结果</span><img :src="absoluteAsset(result.result_image_url)" alt="带检测框结果"></div>
    </div>
    <div class="spacer"></div>
    <div class="grid-2">
      <div class="card"><div class="section-title"><h3>检测详情</h3><a :href="absoluteAsset(result.result_image_url)" download><el-button>保存结果图</el-button></a></div>
        <el-table :data="result.detections" empty-text="未检测到超过阈值的明显异常"><el-table-column prop="class_name_zh" label="类别"/><el-table-column label="置信度"><template #default="s">{{ (s.row.confidence * 100).toFixed(1) }}%</template></el-table-column><el-table-column prop="location_text" label="图像位置"/><el-table-column label="面积"><template #default="s">{{ (s.row.area_ratio * 100).toFixed(2) }}%</template></el-table-column></el-table>
      </div>
      <div class="card"><div class="section-title"><h3>处理建议</h3><router-link :to="`/reports/${result.record_id}`"><el-button type="primary" plain>打开报告</el-button></router-link></div><ol class="advice-list"><li v-for="item in result.advice" :key="item">{{ item }}</li></ol></div>
    </div>
    <div class="spacer"></div><DisclaimerBlock />
  </template>
</template>
