<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api } from '../api'
import MetricCard from '../components/MetricCard.vue'
import RiskTag from '../components/RiskTag.vue'
import DisclaimerBlock from '../components/DisclaimerBlock.vue'
import { useSystemStore } from '../stores/system'
import type { HistoryRecord } from '../types'

const system = useSystemStore()
const recent = ref<HistoryRecord[]>([])
const selectedMetrics = computed(() => system.metrics?.selected?.models?.find((item: any) => item.kind === system.metrics?.selected?.selected_kind))
const summary = computed(() => {
  const items = recent.value.flatMap(item => item.detections || [])
  return {
    total: recent.value.length,
    leak: items.filter(item => item.class_name === 'leak').length,
    crack: items.filter(item => item.class_name === 'crack').length,
    water: items.filter(item => item.class_name === 'water').length,
  }
})
const pct = (value?: number) => value == null ? '—' : `${(value * 100).toFixed(1)}%`

onMounted(async () => {
  await system.refresh()
  recent.value = (await api.get('/history')).data.items.slice(0, 6)
})
</script>

<template>
  <div class="page-header"><div><h2>管道泄漏多目标智能检测系统</h2><p>面向可见光图像的风险筛查、定位与记录管理</p></div><el-tag :type="system.health?.model_available ? 'success' : 'warning'" effect="plain">{{ system.health?.model_available ? `${system.health.device} · 模型就绪` : '最终模型尚未部署' }}</el-tag></div>
  <div class="grid-4">
    <MetricCard label="最近检测记录" :value="summary.total" hint="最多展示最近 200 条" />
    <MetricCard label="疑似泄漏目标" :value="summary.leak" tone="red" />
    <MetricCard label="裂纹目标" :value="summary.crack" tone="orange" />
    <MetricCard label="可见水体目标" :value="summary.water" tone="blue" />
  </div>
  <div class="spacer"></div>
  <div class="card">
    <div class="section-title"><div><h3>当前部署模型指标</h3><small class="muted">仅显示内部测试集真实评估结果</small></div><router-link to="/metrics"><el-button link type="primary">查看完整指标</el-button></router-link></div>
    <div v-if="selectedMetrics" class="grid-4">
      <MetricCard label="Precision" :value="pct(selectedMetrics.precision)" />
      <MetricCard label="Recall" :value="pct(selectedMetrics.recall)" />
      <MetricCard label="mAP50" :value="pct(selectedMetrics.map50)" />
      <MetricCard label="mAP50-95" :value="pct(selectedMetrics.map50_95)" />
    </div>
    <div v-else class="empty-state">尚未完成两组 150 epochs 模型训练与正式评估，冒烟指标不用于客户演示。</div>
  </div>
  <div class="spacer"></div>
  <div class="card">
    <div class="section-title"><h3>最近检测记录</h3><router-link to="/history"><el-button link type="primary">全部记录</el-button></router-link></div>
    <el-table :data="recent" empty-text="暂无检测记录">
      <el-table-column prop="created_at" label="检测时间" width="190" />
      <el-table-column prop="original_filename" label="文件" min-width="180" />
      <el-table-column prop="summary" label="综合状态" min-width="220" />
      <el-table-column label="风险" width="130"><template #default="scope"><RiskTag :level="scope.row.risk_level" /></template></el-table-column>
      <el-table-column label="操作" width="100"><template #default="scope"><router-link :to="`/reports/${scope.row.id}`"><el-button link type="primary">报告</el-button></router-link></template></el-table-column>
    </el-table>
  </div>
  <div class="spacer"></div><DisclaimerBlock />
</template>
