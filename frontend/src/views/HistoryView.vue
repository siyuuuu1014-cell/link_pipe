<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { api, absoluteAsset } from '../api'
import RiskTag from '../components/RiskTag.vue'
import type { HistoryRecord } from '../types'

const records = ref<HistoryRecord[]>([])
const status = ref('')
const risk = ref('')
const day = ref('')
const detail = ref<HistoryRecord | null>(null)
const drawer = ref(false)
async function load() { records.value = (await api.get('/history', { params: { status: status.value || undefined, risk_level: risk.value || undefined, day: day.value || undefined } })).data.items }
async function remove(id: number) { await ElMessageBox.confirm('确认删除这条历史记录？', '删除记录', { type: 'warning' }); await api.delete(`/history/${id}`); ElMessage.success('已删除'); await load() }
function show(row: HistoryRecord) { detail.value = row; drawer.value = true }
onMounted(load)
</script>

<template>
  <div class="page-header"><div><h2>检测历史记录</h2><p>查看、筛选和追溯图片与视频检测</p></div></div>
  <div class="card"><el-form inline><el-form-item label="日期"><el-date-picker v-model="day" type="date" value-format="YYYY-MM-DD" clearable /></el-form-item><el-form-item label="状态"><el-select v-model="status" clearable style="width:190px"><el-option label="疑似泄漏" value="suspected_leak"/><el-option label="相关异常" value="related_anomaly"/><el-option label="未发现明显异常" value="no_visible_anomaly"/></el-select></el-form-item><el-form-item label="风险"><el-select v-model="risk" clearable style="width:160px"><el-option label="高风险" value="high"/><el-option label="中风险" value="medium"/><el-option label="无明显异常" value="none"/></el-select></el-form-item><el-button type="primary" @click="load">筛选</el-button></el-form>
    <el-table :data="records" empty-text="暂无检测记录"><el-table-column prop="created_at" label="时间" width="190"/><el-table-column prop="source_type" label="类型" width="80"/><el-table-column prop="original_filename" label="文件" min-width="180"/><el-table-column prop="summary" label="综合状态" min-width="220"/><el-table-column label="风险" width="130"><template #default="s"><RiskTag :level="s.row.risk_level"/></template></el-table-column><el-table-column label="操作" width="210"><template #default="s"><el-button link type="primary" @click="show(s.row)">详情</el-button><router-link :to="`/reports/${s.row.id}`"><el-button link type="primary">报告</el-button></router-link><el-button link type="danger" @click="remove(s.row.id)">删除</el-button></template></el-table-column></el-table>
  </div>
  <el-drawer v-model="drawer" title="检测详情" size="52%" class="detail-drawer"><template v-if="detail"><div class="grid-2" v-if="detail.source_type === 'image'"><img :src="absoluteAsset(detail.original_image_url)" alt="原图"><img :src="absoluteAsset(detail.result_image_url)" alt="结果图"></div><h3>{{ detail.summary }}</h3><el-table :data="detail.detections"><el-table-column prop="class_name_zh" label="类别"/><el-table-column prop="location_text" label="位置"/><el-table-column label="置信度"><template #default="s">{{ (s.row.confidence*100).toFixed(1) }}%</template></el-table-column></el-table></template></el-drawer>
</template>
