<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { DataAnalysis, Document, Film, House, Picture, Tickets } from '@element-plus/icons-vue'

const route = useRoute()
const active = computed(() => route.path)
const menus = [
  { path: '/', label: '系统总览', icon: House },
  { path: '/detect/image', label: '图片检测', icon: Picture },
  { path: '/detect/video', label: '视频检测', icon: Film },
  { path: '/metrics', label: '模型指标', icon: DataAnalysis },
  { path: '/history', label: '历史记录', icon: Tickets },
]
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-mark">PL</div><div><strong>管道智检</strong><span>PIPELINE VISION</span></div></div>
      <nav>
        <router-link v-for="item in menus" :key="item.path" :to="item.path" :class="{ active: active === item.path }">
          <el-icon><component :is="item.icon" /></el-icon><span>{{ item.label }}</span>
        </router-link>
      </nav>
      <div class="sidebar-foot"><Document /><span>POC 客户演示版本</span></div>
    </aside>
    <main class="main-area">
      <header class="topbar"><div><h1>{{ route.meta.title }}</h1><p>基于 YOLO 的裂纹、可见水体与疑似泄漏定位</p></div><div class="system-chip"><i></i> 本地智能检测</div></header>
      <section class="page-content"><router-view /></section>
    </main>
  </div>
</template>
