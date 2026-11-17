<script setup lang="ts">
import { computed, onMounted } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import MetricCard from '../components/MetricCard.vue'
import { useSystemStore } from '../stores/system'

use([CanvasRenderer, BarChart, GridComponent, LegendComponent, TooltipComponent])
const system = useSystemStore()
onMounted(() => system.refresh())
const models = computed(() => [system.metrics?.full, system.metrics?.lora])
const complete = computed(() => system.metrics?.evaluation_complete)
const pct = (v?: number) => v == null ? '—' : `${(v * 100).toFixed(1)}%`
const chartOption = computed(() => ({
  tooltip: { trigger: 'axis' }, legend: { bottom: 0 }, grid: { left: 45, right: 18, top: 20, bottom: 50 },
  xAxis: { type: 'category', data: ['Precision', 'Recall', 'mAP50', 'mAP50-95'] }, yAxis: { type: 'value', max: 1 },
  series: [
    { name: '完整微调', type: 'bar', data: models.value[0] ? [models.value[0].precision, models.value[0].recall, models.value[0].map50, models.value[0].map50_95] : [] },
    { name: 'LoRA', type: 'bar', data: models.value[1] ? [models.value[1].precision, models.value[1].recall, models.value[1].map50, models.value[1].map50_95] : [] },
  ],
}))
</script>

<template>
  <div class="page-header"><div><h2>模型评估指标</h2><p>完整微调与 LoRA 在内部测试集上的真实结果对比</p></div><el-tag type="info">内部测试集</el-tag></div>
  <el-alert v-if="!complete" title="尚未完成模型评估" description="短轮次冒烟结果不会在此作为正式指标展示；需完成两组 150 epochs 训练后生成最终比较。" type="warning" show-icon :closable="false" />
  <template v-if="models[0] || models[1]"><div class="spacer"></div><div class="grid-2">
    <div v-for="(model, index) in models" :key="index" class="card"><div class="section-title"><h3>{{ index === 0 ? 'YOLO11s 完整微调' : 'YOLO11s + LoRA' }}</h3><el-tag v-if="system.metrics?.selected?.selected_kind === (index === 0 ? 'full' : 'lora')" type="success">已选模型</el-tag></div><div class="grid-2"><MetricCard label="Precision" :value="pct(model?.precision)"/><MetricCard label="Recall" :value="pct(model?.recall)"/><MetricCard label="mAP50" :value="pct(model?.map50)"/><MetricCard label="mAP50-95" :value="pct(model?.map50_95)"/></div><p class="muted">推理 {{ model?.speed_ms?.inference?.toFixed(1) || '—' }} ms · 模型 {{ model?.model_size_mb?.toFixed(1) || '—' }} MB</p></div>
  </div><div class="spacer"></div><div class="card"><h3>核心指标对比</h3><VChart class="chart" :option="chartOption" autoresize /></div>
  <div class="spacer"></div><div class="card"><h3>评估图表</h3><div class="plot-grid"><img :src="'/metrics-assets/full_internal/PR_curve.png'" alt="PR 曲线" @error="($event.target as HTMLImageElement).style.display='none'"><img :src="'/metrics-assets/full_internal/F1_curve.png'" alt="F1 曲线" @error="($event.target as HTMLImageElement).style.display='none'"><img :src="'/metrics-assets/full_internal/confusion_matrix.png'" alt="混淆矩阵" @error="($event.target as HTMLImageElement).style.display='none'"></div></div></template>
  <div v-else class="card empty-state">暂无正式指标文件。系统不会展示虚构指标。</div>
</template>
