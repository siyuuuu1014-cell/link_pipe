<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
const route = useRoute()
const url = computed(() => `/api/v1/reports/${route.params.id}`)
</script>

<template>
  <div class="page-header"><div><h2>检测报告</h2><p>可打印的客户演示报告，包含结果、建议和能力边界</p></div><a :href="url" target="_blank"><el-button type="primary">新窗口打开 / 打印</el-button></a></div>
  <div class="card" style="padding:0;overflow:hidden"><iframe :src="url" title="检测报告" style="width:100%;height:calc(100vh - 180px);border:0" /></div>
</template>
