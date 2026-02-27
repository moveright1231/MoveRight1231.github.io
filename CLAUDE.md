# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Local Development

```bash
bundle exec jekyll serve        # 로컬 서버 실행 (http://localhost:4000)
bundle exec jekyll serve --drafts  # 드래프트 포스트 포함
bundle exec jekyll build        # 정적 파일 빌드 (_site/ 생성)
```

배포는 `git push`로 GitHub Pages가 자동 빌드한다. `_site/`는 빌드 산출물이므로 직접 편집하지 않는다.

## 포스트 작성 규칙

### 파일명 형식
```
_posts/YYYY-MM-DD-title.md
```
날짜는 `YYYY-MM-00` 형식 금지 (Jekyll이 invalid date로 처리할 수 있음).

### 필수 frontmatter
```yaml
---
layout: post
title: 포스트 제목
date: YYYY-MM-DD HH:MM:SS +0800
category: project | paper | experiment
thumbnail: /style/image/{폴더}/{파일}.png   # 반드시 /로 시작하는 절대경로
icon: code | web | ...
---
```

**`thumbnail` 경로는 반드시 `/`로 시작해야 한다.** 상대경로(`style/image/...`)를 쓰면 `/page/2/` 이후 페이지에서 썸네일이 깨진다.

### 이미지 저장 위치
```
style/image/{포스트명}/{파일}.png
```
포스트 내 이미지 참조도 절대경로: `![설명](/style/image/{폴더}/{파일}.png)`

## 아키텍처

```
_config.yml          # Jekyll 설정 (paginate: 12, markdown: kramdown)
_layouts/
  default.html       # 최상위 레이아웃 (header, footer 포함)
  post.html          # 포스트 레이아웃 (사이드 목차 + content)
_includes/
  head.html          # <head> 태그 (CSS/JS 로드, 커스텀 style 오버라이드)
  header.html
  footer.html
  comment.html       # Gitalk 댓글
index.html           # 메인 페이지 (포스트 그리드 + 페이지네이션)
about.md             # About 페이지
style/
  css/style.min.css  # 메인 CSS (minified, 소스 SCSS 없음)
  image/             # 포스트별 이미지 디렉토리
  js/
```

### CSS 커스터마이징
`style.min.css`는 소스 파일 없이 minified만 존재한다. 스타일 수정은 `_includes/head.html` 내 `<style>` 블록으로 오버라이드한다.

현재 오버라이드 내용 (`head.html`):
- 포스트 헤딩의 `#` 장식 마커 제거 (`content: none`)

### 목차 (Contents)
`nav.min.js`가 마크다운의 `#markdown-toc`를 파싱해 사이드바 목차를 자동 생성한다. 목차를 활성화하려면 포스트에 아래를 추가한다:
```markdown
* content
{:toc}
```
